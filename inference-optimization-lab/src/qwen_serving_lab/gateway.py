"""Optional local bounded-admission gateway for vLLM's chat-completions API.

This is a research component, NOT a hardened internet-facing gateway. It does
not implement tenant isolation, billing, global limits, or comprehensive auth.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from .admission import AdmissionLimiter, AdmissionRejected

BACKEND = os.getenv("VLLM_BASE_URL", "http://vllm:8000").rstrip("/")
LIMITER = AdmissionLimiter(
    max_active=int(os.getenv("GATEWAY_MAX_ACTIVE", "8")),
    max_waiting=int(os.getenv("GATEWAY_MAX_WAITING", "16")),
    wait_timeout_s=float(os.getenv("GATEWAY_QUEUE_TIMEOUT_S", "2.0")),
)
API_KEY = os.getenv("GATEWAY_API_KEY", "")
REQUESTS = Counter("gateway_requests_total", "Gateway requests", ["result"])
REJECTIONS = Counter("gateway_admission_rejections_total", "Rejected requests", ["reason"])
WAITS = Histogram("gateway_queue_wait_seconds", "Time spent waiting for admission")
ACTIVE = Gauge("gateway_active_requests", "Requests holding admission slots")
WAITING = Gauge("gateway_waiting_requests", "Requests queued for admission")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(180.0, connect=15.0), trust_env=False
    )
    yield
    await app.state.http.aclose()


app = FastAPI(title="Qwen3 bounded admission gateway", lifespan=lifespan)


def refresh_gauges() -> None:
    ACTIVE.set(LIMITER.active)
    WAITING.set(LIMITER.waiting)


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    refresh_gauges()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions")
async def completion(request: Request) -> Response:
    if API_KEY and request.headers.get("Authorization") != f"Bearer {API_KEY}":
        raise HTTPException(401, "Invalid gateway API key")
    body = await request.json()
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise HTTPException(422, "Expected OpenAI-compatible chat messages")
    try:
        wait = await LIMITER.acquire()
    except AdmissionRejected as exc:
        REJECTIONS.labels(reason=exc.reason).inc()
        REQUESTS.labels(result="rejected").inc()
        refresh_gauges()
        raise HTTPException(429, exc.reason, headers={"Retry-After": "1"}) from exc
    refresh_gauges()
    WAITS.observe(wait)
    http: httpx.AsyncClient = request.app.state.http
    try:
        upstream_request = http.build_request(
            "POST", f"{BACKEND}/v1/chat/completions", json=body
        )
        upstream = await http.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await LIMITER.release()
        refresh_gauges()
        REQUESTS.labels(result="upstream_error").inc()
        raise HTTPException(502, f"Upstream unavailable: {type(exc).__name__}") from exc

    if upstream.status_code >= 400:
        payload = await upstream.aread()
        status = upstream.status_code
        content_type = upstream.headers.get("content-type", "application/json")
        await upstream.aclose()
        await LIMITER.release()
        refresh_gauges()
        REQUESTS.labels(result="upstream_error").inc()
        return Response(payload, status_code=status, media_type=content_type)

    async def release_after_stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
            REQUESTS.labels(result="ok").inc()
        except (httpx.HTTPError, GeneratorExit):
            REQUESTS.labels(result="stream_interrupted").inc()
            raise
        finally:
            await upstream.aclose()
            await LIMITER.release()
            refresh_gauges()

    return StreamingResponse(
        release_after_stream(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )
