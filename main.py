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

    # tiny config - MHA
    B = 1
    Hkv = 8
    Hq = 8
    D = 64
    T = 128
    dtype = torch.float16
    device="cuda"

    S = [256, 512, 1024, 4096, 8192]

    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)
    print()

    print("Multi-head attention")
    for i in S:
        torch.cuda.reset_peak_memory_stats()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        k_cache = torch.randn((B, Hkv, i, D), device=device, dtype=dtype)
        v_cache = torch.randn((B, Hkv, i, D), device=device, dtype=dtype) 
        kvbytes = ((k_cache.numel() + v_cache.numel()) * k_cache.element_size())

        lonely_q = torch.randn((B, Hq, 1, D), device=device, dtype=dtype) 

        # warmup
        scores = torch.matmul(lonely_q, k_cache.transpose(2,3))

        torch.cuda.synchronize()
        start.record()
        for decode_t in range(T):
            scores = torch.matmul(lonely_q, k_cache.transpose(2,3))
            scaled_logits = scores / (D ** 0.5)
            awts = torch.nn.functional.softmax(scaled_logits, dim=-1)
            Z = torch.matmul(awts, v_cache)
        end.record()
        end.synchronize()

        peak_alloc = torch.cuda.max_memory_allocated()

        time_tot = start.elapsed_time(end)
        time_unit = time_tot / T;

        print(
            f"i={i:5d} k={tuple(k_cache.shape)}  "
            f"KV={kvbytes/1e6:7.2f}MB peak_alloc={peak_alloc/1e6:7.2f}MB"
            )

        print(
            f"scores={tuple(scores.shape)} awts={tuple(awts.shape)}  "
            )

        print(
                f"Execution time: {time_unit:0.3f} ms / tok"
            )

    # tiny config - MQA
    B = 1
    Hkv = 1
    Hq = 8
    D = 64
    T = 128
    dtype = torch.float16
    device="cuda"

    S = [256, 512, 1024, 4096, 8192]

    print("Multi-Query Attention")
    for i in S:
        torch.cuda.reset_peak_memory_stats()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        k_cache = torch.randn((B, Hkv, i, D), device=device, dtype=dtype)
        v_cache = torch.randn((B, Hkv, i, D), device=device, dtype=dtype) 
        kvbytes = ((k_cache.numel() + v_cache.numel()) * k_cache.element_size())

        lonely_q = torch.randn((B, Hq, 1, D), device=device, dtype=dtype) 

        # warmup
        scores = torch.matmul(lonely_q, k_cache.transpose(2,3))

        torch.cuda.synchronize()
        start.record()
        for decode_t in range(T):
            scores = torch.matmul(lonely_q, k_cache.transpose(2,3))
            scaled_logits = scores / (D ** 0.5)
            awts = torch.nn.functional.softmax(scaled_logits, dim=-1)
            Z = torch.matmul(awts, v_cache)
        end.record()
        end.synchronize()

        peak_alloc = torch.cuda.max_memory_allocated()

        time_tot = start.elapsed_time(end)
        time_unit = time_tot / T;

        print(
            f"i={i:5d} k={tuple(k_cache.shape)}  "
            f"KV={kvbytes/1e6:7.2f}MB peak_alloc={peak_alloc/1e6:7.2f}MB"
            )

        print(
            f"scores={tuple(scores.shape)} awts={tuple(awts.shape)}  "
            )

        print(
                f"Execution time: {time_unit:0.3f} ms / tok"
            )

    # tiny config - GQA
    B = 1
    Hkv = 4
    Hq = 8
    D = 64
    T = 128
    dtype = torch.float16
    device="cuda"

    S = [256, 512, 1024, 4096, 8192]

    print("Grouped Query Attention")
    for i in S:
        torch.cuda.reset_peak_memory_stats()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        k_cache = torch.randn((B, Hkv, i, D), device=device, dtype=dtype)
        v_cache = torch.randn((B, Hkv, i, D), device=device, dtype=dtype) 
        kvbytes = ((k_cache.numel() + v_cache.numel()) * k_cache.element_size())

        lonely_q = torch.randn((B, Hq, 1, D), device=device, dtype=dtype)

        k_exp = k_cache.repeat_interleave(Hq // Hkv, dim=1)
        v_exp = v_cache.repeat_interleave(Hq // Hkv, dim=1)

        # warmup
        scores = torch.matmul(lonely_q, k_exp.transpose(2,3))

        torch.cuda.synchronize()
        start.record()
        for decode_t in range(T):
            scores = torch.matmul(lonely_q, k_exp.transpose(2,3))
            scaled_logits = scores / (D ** 0.5)
            awts = torch.nn.functional.softmax(scaled_logits, dim=-1)
            Z = torch.matmul(awts, v_exp)
        end.record()
        end.synchronize()

        peak_alloc = torch.cuda.max_memory_allocated()

        time_tot = start.elapsed_time(end)
        time_unit = time_tot / T;

        print(
            f"i={i:5d} k={tuple(k_cache.shape)}  "
            f"KV={kvbytes/1e6:7.2f}MB peak_alloc={peak_alloc/1e6:7.2f}MB"
            )

        print(
            f"scores={tuple(scores.shape)} awts={tuple(awts.shape)}  "
            )

        print(
                f"Execution time: {time_unit:0.3f} ms / tok"
            )

if __name__ == "__main__":
    main()
