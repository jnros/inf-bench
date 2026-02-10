# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GPU inference microbenchmark. Measures KV-cache attention (QK^T matmul) scaling
across sequence lengths. Single-file project: `main.py`.

## Run

```
uv run main.py    # requires CUDA GPU
```

## Environment

- Python 3.11 via uv, venv at `.venv/`
- PyTorch 2.7.1+cu118 (torch, torchvision, torchaudio)
- Disables flash/mem-efficient SDP; forces math SDP for controlled measurement

## Architecture

`main.py` runs a loop over sequence lengths [256..8192]. For each:
1. Allocates KV cache tensors `(B,H,S,D)` and a single query `(B,H,1,D)`
2. Warmup matmul + sync
3. Timed `Q @ K^T` via CUDA events
4. Reports KV size, peak alloc, execution time

Config: B=1, H=1, D=64, float16. No CLI args yet.
