import math
import time
import torch

# Force PyTorch to NOT silently swap in flash/mem-efficient attention kernels.
# We want our own codepath to be the thing we're measuring.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required for benchmark"

    # tiny config
    B = 1
    H = 1
    D = 64
    dtype = torch.float16
    device="cuda"

    S = [256, 512, 1024, 4096, 8192]

    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)
    print()

    for i in S:
        torch.cuda.reset_peak_memory_stats()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        k_cache = torch.empty((B, H, i, D), device=device, dtype=dtype) 
        v_cache = torch.empty((B, H, i, D), device=device, dtype=dtype) 
        kvbytes = ((k_cache.numel() + v_cache.numel()) * k_cache.element_size())

        lonely_q = torch.empty((B, H, 1, D), device=device, dtype=dtype) 

        # warmup
        scores = torch.matmul(lonely_q, k_cache.transpose(2,3))
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        start.record()
        scores = torch.matmul(lonely_q, k_cache.transpose(2,3))
        end.record()
        end.synchronize()

        peak_alloc = torch.cuda.max_memory_allocated()

        print(
            f"i={i:5d} k={tuple(k_cache.shape)}  "
            f"KV={kvbytes/1e6:7.2f}MB peak_alloc={peak_alloc/1e6:7.2f}MB"
            )

        print(
            f"scores={tuple(scores.shape)}  "
            )

        print(
                f"Execution time: {((start.elapsed_time(end))):0.3f} ms"
            )

if __name__ == "__main__":
    main()
