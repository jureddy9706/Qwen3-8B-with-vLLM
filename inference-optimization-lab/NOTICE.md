# Attribution and licensing

The Python orchestration, prompt generator, bounded-admission gateway, summary
calculations, experiment comparator, and tests in this repository were written
for this project and are licensed under MIT (see LICENSE).

This repository does **not** contain Qwen3 model weights or vLLM source code.
It integrates independently installed upstream dependencies:

- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B): open-weight model from Qwen/Alibaba, Apache-2.0, separately licensed.
- [vLLM](https://github.com/vllm-project/vllm): inference engine, Apache-2.0, separately licensed.
- [FastAPI](https://fastapi.tiangolo.com/), [httpx](https://www.python-httpx.org/),
  [Prometheus](https://prometheus.io/): third-party libraries and services with their own licenses.

This implementation follows public API documentation, not copied code from a
third-party inference-benchmarking repository. Reported performance belongs to
those respective authors unless independently reproduced here.
