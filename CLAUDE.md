# CLAUDE.md

## Run

```
uv run main.py      # bench on local GPU → results/{gpu-slug}.csv + .png
uv run compare.py   # cross-GPU comparison → results/compare.png
```

## Environment

- Python 3.11, uv, venv at `.venv/`
- PyTorch 2.7.1+cu118, matplotlib
- Forces math SDP only (no flash/mem-efficient)

## Architecture

`CONFIGS` table: MHA (8,8), MQA (1,8), GQA (4,8) as (Hkv,Hq).

`bench()` per sequence length [256..8192] (extended to 65536 on >=24GB VRAM):
1. Alloc KV cache `(B,Hkv,S,D)` + query `(B,Hq,1,D)`
2. GQA: `repeat_interleave` KV heads; MQA: broadcast; MHA: noop
3. Warmup + sync
4. Timed loop (T iters): Q@K^T, scale, softmax, awts@V
5. Compute effective bandwidth: `bytes_touched / ms_tok`
6. Return list of dicts; report KV size, peak alloc, ms/tok

`plot()` — 3-panel figure: ms/tok, KV cache MB, bandwidth GB/s vs S.

`compare.py` — reads all `results/*.csv`, plots MHA-only cross-GPU
comparison with theoretical peak bandwidth lines.

Defaults: B=1, D=64, T=128, float16. No CLI args.
Output: `results/{gpu-slug}.csv`, `results/{gpu-slug}.png`, `results/compare.png`.
