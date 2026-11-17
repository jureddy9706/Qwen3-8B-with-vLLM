from dataclasses import replace

import pytest

from qwen_serving_lab.core import Workload, make_prompts, percentile, summarize
from qwen_serving_lab.bench import parse_sse_line
from qwen_serving_lab.compare import compare


def test_prompts_reproducible_and_variable():
    cfg = Workload(num_requests=12, min_prompt_words=2, max_prompt_words=10)
    assert make_prompts(cfg) == make_prompts(cfg)
    assert make_prompts(cfg) != make_prompts(replace(cfg, seed=42))
    assert len(set(make_prompts(cfg))) == 12


def test_percentile_and_slo_goodput():
    rows = [
        {"ok": True, "ttft_ms": 100.0, "e2e_ms": 800.0, "client_wait_ms": 0.0, "output_tokens": 10},
        {"ok": True, "ttft_ms": 3000.0, "e2e_ms": 3200.0, "client_wait_ms": 10.0, "output_tokens": 10},
        {"ok": False, "ttft_ms": None, "e2e_ms": 30.0, "client_wait_ms": 0.0, "output_tokens": None},
    ]
    summary = summarize(rows, 5.0, Workload(ttft_slo_ms=1000))
    assert summary["slo_passed"] == 1
    assert summary["slo_goodput_rps"] == 0.2
    assert summary["completed_rps"] == 0.4
    assert summary["output_tokens_per_second"] == 4.0
    assert percentile([], 0.95) is None


def test_sse_usage_and_visible_content():
    assert parse_sse_line('data: {"choices":[{"delta":{"content":"hello"}}]}')["choices"][0]["delta"]["content"] == "hello"
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line(": heartbeat") is None


def test_compare_requires_matched_workloads():
    a = {"schema_version": 1, "fingerprint": {"seed": 41}, "summary": {"slo_goodput_rps": 2.0}}
    b = {"schema_version": 1, "fingerprint": {"seed": 41}, "summary": {"slo_goodput_rps": 3.0}}
    assert compare(a, b)["changes"]["slo_goodput_rps"]["relative_change_pct"] == 50.0
    b["fingerprint"]["seed"] = 42
    with pytest.raises(ValueError, match="identical"):
        compare(a, b)


def test_streaming_http_response_parsing():
    import asyncio
    import httpx
    from qwen_serving_lab.bench import one_request

    async def check():
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            assert b'enable_thinking' in request.content
            stream = (
                b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
                b'data: [DONE]\n\n'
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream)

        cfg = Workload(num_requests=1, min_prompt_words=1, max_prompt_words=2)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            row = await one_request(client, cfg, 0, "example", asyncio.Semaphore(1), 0.0)
        assert row["ok"]
        assert row["output_tokens"] == 1
        assert row["ttft_ms"] >= 0
        assert row["http_status"] == 200

    asyncio.run(check())
