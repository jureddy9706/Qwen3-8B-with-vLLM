# Experimental results and evidence

## Baseline already in the parent repository (existing material)

The parent repository already contains a Qwen3-8B / vLLM 0.30.0 single-A10G benchmark with a table in [../bench_summary.txt](../bench_summary.txt) and a plot in [../throughput_vs_ttft.png](../throughput_vs_ttft.png).

| Concurrency | Existing output tokens/s | Existing P99 TTFT |
| ---: | ---: | ---: |
| 1 | 28.74 | 150.61 ms |
| 5 | 128.64 | 684.97 ms |
| 20 | 385.59 | 2518.70 ms |
| 40 | 506.61 | 9626.13 ms |

These numbers predate this extension. Their benchmark method and workload differ from the new deterministic, arrival-rate-driven gateway comparison, so **do not compare these numbers directly against new runs or claim a speedup**. Their provenance is whatever was recorded in the original benchmark; we have not independently reproduced them here.

## Optimization lab validation (new work)

- CPU unit/HTTP mock tests: **9 passed** when invoked with `PYTHONPATH=src pytest -q`.
- Source compile check: passed.
- GPU Qwen3-8B inference using the Docker Compose setup: **not run in this environment** (no NVIDIA GPU available).
- Gateway versus direct-vLLM performance improvement: **not measured**.
- Cost-per-token, SLO goodput change, and model-quality preservation: **not measured**.

## Reproduce and capture your own results on an NVIDIA GPU

From this `inference-optimization-lab/` directory:

```bash
python -m pip install -e '.[dev]'
pytest -q
docker compose up -d --build
# Wait for model initialization; inspect docker compose logs -f vllm
qwen-lab-bench --config configs/workload.yaml --base-url http://127.0.0.1:8000 --output results/direct.json
qwen-lab-bench --config configs/workload.yaml --base-url http://127.0.0.1:8080 --output results/gateway.json
qwen-lab-compare --baseline results/direct.json --candidate results/gateway.json --output results/comparison.json
```

Run both endpoints under the *same offered load* and hardware, with warmup and repeated independent trials. Record GPU model, vLLM image digest, model/tokenizer revision, effective prompt token lengths, server metrics, rejection rates, and each trial's JSON traces. Only then report p95/p99 TTFT, completed RPS, SLO-goodput, and rejection trade-offs. A gateway can reduce tail latency by shedding load while **reducing overall completion throughput**; report both.

## Experiment table (pending GPU validation)

| Workload rate | Baseline SLO-goodput | Gateway SLO-goodput | Baseline P99 TTFT | Gateway P99 TTFT | 429 rate | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 req/s | Pending | Pending | Pending | Pending | Pending | Pending |
| 1 req/s | Pending | Pending | Pending | Pending | Pending | Pending |
| 2 req/s | Pending | Pending | Pending | Pending | Pending | Pending |
| 4 req/s | Pending | Pending | Pending | Pending | Pending | Pending |

No improvement is claimed until observed in replicated runs.
