# Attribution and licensing

The Python orchestration, prompt generator, bounded-admission gateway, summary
calculations, experiment comparator, and tests in this repository are licensed under MIT (see LICENSE).

This repository does **not** contain Qwen3 model weights or vLLM source code.
It integrates independently installed upstream dependencies:

- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B): open-weight model from Qwen/Alibaba, Apache-2.0, separately licensed.
- [vLLM](https://github.com/vllm-project/vllm): inference engine, Apache-2.0, separately licensed.
- [FastAPI](https://fastapi.tiangolo.com/), [httpx](https://www.python-httpx.org/),
  [Prometheus](https://prometheus.io/): third-party libraries and services with their own licenses.

This implementation uses documented upstream APIs. Benchmark results are attributed
to their respective experiments; measurements from different workloads are not directly comparable.
