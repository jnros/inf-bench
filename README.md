# inf-bench

GPU microbenchmark: KV-cache attention (MHA, MQA, GQA) across sequence lengths.
Measures effective memory bandwidth to show why decode is memory-bound.

## Usage

```
uv run main.py      # requires CUDA GPU, Python 3.11+, uv
uv run compare.py   # after collecting results from multiple GPUs
```

Outputs (per GPU):
- `results/{gpu-slug}.csv` — structured data for all runs
- `results/{gpu-slug}.png` — 3-panel chart (ms/tok, KV cache MB, bandwidth GB/s)

Cross-GPU comparison:
- `results/compare.png` — MHA-only overlay with theoretical peak BW lines

Seqlens extend to 65536 automatically on GPUs with >=24GB VRAM.
