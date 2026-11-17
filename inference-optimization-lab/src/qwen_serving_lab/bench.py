"""Open-loop HTTP streaming benchmark with trace-level TTFT and SLO goodput.

TTFT is measured to the first non-empty *visible content* delta, not to the
HTTP response headers. Client admission wait is measured separately.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any

import httpx

from .core import Workload, make_prompts, summarize


def parse_sse_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    raw = line[5:].strip()
    if not raw or raw == "[DONE]":
        return None
    return json.loads(raw)


async def one_request(
    client: httpx.AsyncClient, spec: Workload, sample: int, prompt: str,
    sem: asyncio.Semaphore, scheduled_at: float,
) -> dict[str, Any]:
    """Send one streaming chat completion. Errors are recorded, not swallowed."""
    async with sem:
        start = time.perf_counter()
        result: dict[str, Any] = {
            "sample": sample,
            "client_wait_ms": round((start - scheduled_at) * 1000, 3),
            "ttft_ms": None, "e2e_ms": None, "output_tokens": None,
            "ok": False, "http_status": None, "error": None,
        }
        body = {
            "model": spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": spec.max_output_tokens,
            "temperature": 0.7,
            "seed": spec.seed + sample,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": spec.enable_thinking},
        }
        saw_content = False
        try:
            async with client.stream(
                "POST", f"{spec.base_url.rstrip('/')}/v1/chat/completions", json=body
            ) as response:
                result["http_status"] = response.status_code
                response.raise_for_status()
                async for line in response.aiter_lines():
                    packet = parse_sse_line(line)
                    if packet is None:
                        continue
                    if isinstance(packet.get("usage"), dict):
                        result["output_tokens"] = packet["usage"].get("completion_tokens")
                    for choice in packet.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if content and not saw_content:
                            result["ttft_ms"] = round((time.perf_counter() - start) * 1000, 3)
                            saw_content = True
            if not saw_content:
                raise ValueError("Stream ended without visible content")
            result["ok"] = True
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            result["error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
        finally:
            result["e2e_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return result


async def run(spec: Workload) -> dict[str, Any]:
    prompts = make_prompts(spec)
    sem = asyncio.Semaphore(spec.max_inflight)
    started = time.perf_counter()
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    timeout = httpx.Timeout(spec.timeout_seconds, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        for i, prompt in enumerate(prompts):
            scheduled = started + i / spec.arrival_rps
            delay = scheduled - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one_request(client, spec, i, prompt, sem, scheduled)))
        rows = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": {"python": platform.python_version(), "httpx": httpx.__version__},
        "workload": asdict(spec), "fingerprint": spec.fingerprint(),
        "summary": summarize(rows, elapsed, spec), "requests": rows,
        "notes": [
            "Prompts are specified in words, not tokenizer-measured tokens.",
            "TTFT is first visible streamed content, excluding client-side queue wait.",
            "Output token rate is null unless usage is reported for every successful request.",
            "These are real HTTP measurements only if the configured endpoint was running.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="YAML workload specification")
    p.add_argument("--output", required=True, help="Where to save raw trace + summary JSON")
    p.add_argument("--base-url", default=None, help="Override API endpoint")
    p.add_argument("--dry-run", action="store_true", help="Validate and show config without requests")
    args = p.parse_args()
    spec = Workload.load(args.config)
    if args.base_url:
        spec = replace(spec, base_url=args.base_url)
    if args.dry_run:
        print(json.dumps(asdict(spec), indent=2))
        return
    data = asyncio.run(run(spec))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data["summary"], indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
