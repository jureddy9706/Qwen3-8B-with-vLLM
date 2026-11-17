"""Experiment configuration, reproducible prompt mixes, and percentile summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from typing import Any

import yaml


@dataclass(frozen=True)
class Workload:
    model: str = "Qwen/Qwen3-8B"
    base_url: str = "http://127.0.0.1:8000"
    num_requests: int = 100
    arrival_rps: float = 2.0
    max_inflight: int = 16
    seed: int = 41
    max_output_tokens: int = 64
    timeout_seconds: float = 120.0
    min_prompt_words: int = 100
    max_prompt_words: int = 600
    shared_prefix: bool = True
    ttft_slo_ms: float = 1500.0
    e2e_slo_ms: float = 12000.0
    enable_thinking: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "Workload":
        obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("Expected a YAML mapping")
        unknown = set(obj) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown workload options: {sorted(unknown)}")
        spec = cls(**obj)
        if not spec.model or not spec.base_url.startswith(("http://", "https://")):
            raise ValueError("model and an HTTP(S) base_url are required")
        if not (1 <= spec.num_requests <= 10000 and 1 <= spec.max_inflight <= 1024):
            raise ValueError("num_requests or max_inflight out of range")
        if spec.arrival_rps <= 0 or spec.max_output_tokens < 1:
            raise ValueError("arrival_rps and max_output_tokens must be positive")
        if spec.timeout_seconds <= 0 or spec.ttft_slo_ms <= 0 or spec.e2e_slo_ms <= 0:
            raise ValueError("timeouts and SLOs must be positive")
        if not (1 <= spec.min_prompt_words <= spec.max_prompt_words <= 10000):
            raise ValueError("Invalid prompt word range")
        return spec

    def fingerprint(self) -> dict[str, Any]:
        """Fields that must match to make an apples-to-apples experiment comparison."""
        fields = (
            "model", "num_requests", "arrival_rps", "max_inflight", "seed",
            "max_output_tokens", "min_prompt_words", "max_prompt_words", "shared_prefix",
            "ttft_slo_ms", "e2e_slo_ms", "enable_thinking",
        )
        data = asdict(self)
        return {key: data[key] for key in fields}


VOCABULARY = (
    "customer", "support", "policy", "shipment", "invoice", "account", "request",
    "status", "refund", "priority", "history", "issue", "delivery", "resolve", "context",
    "service", "workflow", "data", "summary", "routing", "quality", "response",
)
COMMON_PREFIX = (
    "You are a customer-service analyst. Read the following synthetic business notes. "
    "Summarize the request in one sentence; do not invent facts. "
    "The notes contain no private or real customer data.\n"
)


def make_prompts(spec: Workload) -> list[str]:
    """Generate deterministic *word-length* approximations, not guaranteed token lengths."""
    rng = random.Random(spec.seed)
    prompts = []
    for i in range(spec.num_requests):
        words = rng.randint(spec.min_prompt_words, spec.max_prompt_words)
        text = " ".join(rng.choices(VOCABULARY, k=words))
        prefix = COMMON_PREFIX if spec.shared_prefix else f"Analyze synthetic record {i}.\n"
        prompts.append(f"{prefix}Record {i}: {text}\nSummary:")
    return prompts


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo), 3)


def summarize(rows: list[dict[str, Any]], elapsed_s: float, spec: Workload) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    e2e = [r["e2e_ms"] for r in ok if r.get("e2e_ms") is not None]
    waits = [r["client_wait_ms"] for r in rows if r.get("client_wait_ms") is not None]
    tokens = [r["output_tokens"] for r in ok if r.get("output_tokens") is not None]
    good = sum(
        r.get("ok", False)
        and r.get("ttft_ms") is not None
        and r["ttft_ms"] <= spec.ttft_slo_ms
        and r["e2e_ms"] <= spec.e2e_slo_ms
        for r in rows
    )
    elapsed = max(elapsed_s, 1e-9)
    return {
        "sent": len(rows), "success": len(ok), "failed": len(rows) - len(ok),
        "slo_passed": good, "slo_goodput_rps": round(good / elapsed, 4),
        "completed_rps": round(len(ok) / elapsed, 4),
        "p50_ttft_ms": percentile(ttft, 0.50), "p95_ttft_ms": percentile(ttft, 0.95),
        "p99_ttft_ms": percentile(ttft, 0.99), "p95_e2e_ms": percentile(e2e, 0.95),
        "p95_client_wait_ms": percentile(waits, 0.95),
        "output_tokens_per_second": round(sum(tokens) / elapsed, 2)
        if len(tokens) == len(ok) and len(ok) > 0 else None,
        "token_usage_coverage": f"{len(tokens)}/{len(ok)}",
        "elapsed_seconds": round(elapsed_s, 3),
    }
