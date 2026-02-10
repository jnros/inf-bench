# CLAUDE.md

## Run

```
uv run main.py    # requires CUDA GPU
```

## Environment

- Python 3.11, uv, venv at `.venv/`
- PyTorch 2.7.1+cu118
- Forces math SDP only (no flash/mem-efficient)

## Architecture

`CONFIGS` table: MHA (8,8), MQA (1,8), GQA (4,8) as (Hkv,Hq).

`bench()` per sequence length [256..8192]:
1. Alloc KV cache `(B,Hkv,S,D)` + query `(B,Hq,1,D)`
2. GQA: `repeat_interleave` KV heads; MQA: broadcast; MHA: noop
3. Warmup + sync
4. Timed loop (T iters): Q@K^T, scale, softmax, awts@V
5. Report KV size, peak alloc, ms/tok

Defaults: B=1, D=64, T=128, float16. No CLI args.
