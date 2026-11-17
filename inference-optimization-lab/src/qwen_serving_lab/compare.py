"""Strictly paired comparisons of real benchmark traces; never fabricate results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if a.get("schema_version") != 1 or b.get("schema_version") != 1:
        raise ValueError("Unsupported result schema")
    if a.get("fingerprint") != b.get("fingerprint"):
        raise ValueError("Experiments must have identical model, prompts, rate, concurrency, and SLOs")
    baseline, candidate = a["summary"], b["summary"]
    fields = (
        "p95_ttft_ms", "p99_ttft_ms", "p95_e2e_ms", "p95_client_wait_ms",
        "completed_rps", "slo_goodput_rps", "output_tokens_per_second",
    )
    changes = {}
    for key in fields:
        old, new = baseline.get(key), candidate.get(key)
        changes[key] = {
            "baseline": old, "candidate": new,
            "relative_change_pct": round((new / old - 1) * 100, 2)
            if isinstance(old, (int, float)) and old != 0 and isinstance(new, (int, float))
            else None,
        }
    return {
        "pairing": "matched configuration (not necessarily statistically significant)",
        "changes": changes,
        "caution": "Repeat independent runs, report variance, and distinguish SLO-goodput from raw throughput.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    a = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    b = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    data = compare(a, b)
    report = json.dumps(data, indent=2) + "\n"
    print(report)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
