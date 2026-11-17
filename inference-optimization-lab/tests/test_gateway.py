import asyncio
import httpx

from qwen_serving_lab.admission import AdmissionLimiter
from qwen_serving_lab import gateway


def test_gateway_proxies_stream_and_releases_slot(monkeypatch):
    async def check():
        monkeypatch.setattr(gateway, "LIMITER", AdmissionLimiter(1, 0, 0.01))
        async def backend(_: httpx.Request) -> httpx.Response:
            class S(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield b"data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=S())
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as backend_client:
            gateway.app.state.http = backend_client
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "test"}], "stream": True})
                assert response.status_code == 200
                assert "[DONE]" in response.text
                assert gateway.LIMITER.active == 0
                await gateway.LIMITER.acquire()
                rejected = await client.post("/v1/chat/completions", json={"messages": []})
                assert rejected.status_code == 429
                await gateway.LIMITER.release()
    asyncio.run(check())
