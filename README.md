# inf-bench

**WORK IN PROGRESS** — everything here is experimental and subject to change.

GPU microbenchmark for KV-cache attention scaling. Full decode attention
(Q@K^T, scale, softmax, awts@V) timed across sequence lengths.

## Requirements

- CUDA GPU
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

## Usage

```
uv run main.py
```
