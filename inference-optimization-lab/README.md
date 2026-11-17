# Qwen3-8B Inference Optimization Lab

**Reproducible inference experiments and a bounded-admission gateway for open-weight LLMs.**

This project combines a Dockerized [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) / [vLLM](https://github.com/vllm-project/vllm) deployment, synthetic workload replay, request-level latency traces, SLO goodput analysis, Prometheus metrics, and a small admission-control gateway. It is designed as an engineering experiment, **not a benchmark claiming unmeasured speedups**.

> **Status (2026-11):** Source code and CPU-based unit tests are included. GPU-backed Qwen3-8B execution, real latency measurements, and throughput improvements have **not yet been validated or claimed** in this repository. Run the experiment and commit your own traces before citing metrics in a resume.

## Existing A10G benchmark vs. this extension

The repository root contains earlier Qwen3-8B A10G concurrency measurements and `bench_summary.txt` (including approximately 28.7 output tok/s at concurrency 1 and 385.6 tok/s at concurrency 20). **These are pre-existing baseline results; they are not measurements of this gateway.** See [RESULTS.md](RESULTS.md) for separate validation and upcoming optimization trials.

## Architecture

```text
Synthetic deterministic prompt generator
               |
       Open-loop load generator ------> Trace JSON + SLO analysis
               |
        baseline / candidate
               |
        +------+-------+
        |              |
    direct :8000    gateway :8080  <-- bounded queue, 429, wait time
        |              |
        +------+-------+
               |
          vLLM V1
         Qwen3-8B
               |
  /metrics --> Prometheus :9090
```

The baseline uses vLLM directly. The candidate routes the **same load** through a gateway that caps active work, bounds the waiting room, and returns 429 when overloaded. It is an experiment in **load shedding and tail latency**, not a guarantee of higher total throughput. Benchmark end-to-end request success and SLO goodput; evaluate trade-offs in rejection rate.

## Requirements

- Linux with an NVIDIA GPU, compatible NVIDIA driver, and NVIDIA Container Toolkit for Docker GPU support. Qwen3-8B BF16 weights alone occupy roughly 16.4 GB; KV cache and runtime overhead require additional memory. A 24 GB card may work with conservative context/concurrency; more memory is preferable.
- Docker Compose v2 with GPU support; Python 3.10+ for the local benchmark client.
- Model download access and enough disk space. Hugging Face credentials may be needed in some environments. Do **not** commit tokens or weights.

## Quick start

```bash
cp .env.example .env
python -m pip install -e '.[dev]'
pytest -q
python -m qwen_serving_lab.bench --config configs/workload.yaml --output results/dry-run.json --dry-run

docker compose up -d --build
# Wait for model startup; inspect `docker compose logs -f vllm`.
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8000/metrics
```

`VLLM_IMAGE` defaults to a moving image tag for ease of setup. **Pin a tested image digest** in `.env` and record it with your results for research-grade reproducibility. The project does not host the model files itself.

## Run a matched comparison

```bash
# 1. Baseline: direct vLLM endpoint.
qwen-lab-bench --config configs/workload.yaml --base-url http://127.0.0.1:8000 --output results/baseline.json

# 2. Candidate: same prompts, arrival rate, concurrency, and SLO settings.
qwen-lab-bench --config configs/workload.yaml --base-url http://127.0.0.1:8080 --output results/gateway.json

# 3. Strict pairing guard checks workload fingerprint.
qwen-lab-compare --baseline results/baseline.json --candidate results/gateway.json --output results/comparison.json
```

Modify `configs/workload.yaml` to sweep arrival rates (e.g. 0.5, 1, 2, 4 requests/s), request counts, prompt mix and output limits. **Record each rate as a separate baseline/candidate pair**, rather than comparing different offered loads. Re-run each configuration several times after warmup and report variability. The generated notes are synthetic and should not be represented as customer data.

For raw vLLM cross-checks, also use its maintained `vllm bench serve` command:

```bash
# Run inside a suitable vLLM environment; inspect version-specific flags.
vllm bench serve --help
```

## Metrics that matter

| Signal | Definition / caveat |
|---|---|
| `p95_ttft_ms`, `p99_ttft_ms` | First **non-empty visible content** SSE delta after request dispatch. Excludes client-side semaphore wait; not equivalent to HTTP time-to-first-byte. |
| `p95_client_wait_ms` | Load-generator-side wait from scheduled arrival to dispatch. This may grow when the client reaches `max_inflight`. |
| `p95_e2e_ms` | Request dispatch through end of HTTP streamed response. |
| `completed_rps` | Successful requests divided by **total experiment wall time**. |
| `slo_goodput_rps` | Successful requests meeting both configured per-request TTFT and E2E thresholds, divided by wall time. |
| `output_tokens_per_second` | Only reported if server-supplied usage includes completion token counts for **all** successful requests; otherwise `null`. |
| `gateway_admission_rejections_total` | Immediate queue overflow and wait timeouts, reported separately from transport errors. |

Prompt length is specified in *words*, not verified tokenizer tokens. For exact token-length experiments, extend the generator to use the model tokenizer. Qwen3 runs with `enable_thinking: false` by default so visible-output TTFT is comparable across runs. Always document this switch; changing thinking mode invalidates paired comparisons.

Inspect real server metrics at `http://127.0.0.1:9090` (Prometheus query UI). Useful upstream series include `vllm:num_requests_running`, `vllm:kv_cache_usage_perc`, and `vllm:prefix_cache_hits`; metric availability may vary with vLLM version. Gateway series include `gateway_active_requests`, `gateway_waiting_requests`, `gateway_queue_wait_seconds`, and admission rejection counters.

## Suggested investigations

1. **Queueing under bursty load:** Hold model and arrival rate fixed; change `GATEWAY_MAX_ACTIVE`, `GATEWAY_MAX_WAITING`, and `GATEWAY_QUEUE_TIMEOUT_S`. Optimize SLO goodput and report 429 costs.
2. **Prefix reuse:** Re-run a matched sweep with `shared_prefix` on and off, then inspect vLLM prefix-cache counters. Note that toggling the prompt set changes the workload, so this is a separate study, not a matched before/after pair.
3. **Serving configuration:** Change vLLM `--max-num-seqs` or model precision in **separate deployment configurations**. Document exact hardware, model revision, image digest, and output-quality check.
4. **Quality-performance trade-off:** Add a task-specific held-out evaluation corpus before making any claim that a quantized or optimized configuration preserves quality.

## Reproducibility and limitations

- No pretrained weights, private training data, or third-party benchmark numbers are included.
- The gateway is **per-process only**, not a distributed global limiter, and is **not production security infrastructure**. It does not provide customer/tenant isolation, authentication policy enforcement, abuse prevention, or durable queues. Keep ports bound to localhost unless these controls are added.
- Model startup time, GPU memory pressure, preemption, tokenizer revision, prompt tokens, vLLM version, CUDA version, load-generator capacity, warmup, and server-side queue metrics affect measured performance. Record these alongside trace JSON.
- Current measurements record client-side end-to-end behavior and vLLM's exposed Prometheus metrics. Streaming intermediate content chunks are **not assumed to equal tokens**, and this tool therefore does not mislabel chunk gaps as TPOT.
- The run comparator validates identical workload settings, but a matched config is **not** a statistically controlled experiment by itself.
- This repository's code is new, built against documented upstream APIs. See [NOTICE.md](NOTICE.md) for upstream attribution and independent licenses.

## Portfolio / resume usage

Safe description **after you complete and validate a GPU experiment**:

> Built an open-source Qwen3-8B inference testbed with Docker/vLLM serving, deterministic workload replay, bounded admission control, Prometheus telemetry, and paired SLO-goodput comparisons; evaluated queueing and KV-cache behavior across concurrency sweeps.

Only append a quantified outcome after collecting real baseline/candidate traces. Do not present this independent project as work performed at a prior employer.

## References

- [Official Qwen3-8B model card (Apache-2.0)](https://huggingface.co/Qwen/Qwen3-8B)
- [vLLM CLI and benchmark guide](https://docs.vllm.ai/en/latest/cli/)
- [vLLM metrics documentation](https://docs.vllm.ai/en/latest/design/metrics/)
- [vLLM source repository](https://github.com/vllm-project/vllm)
