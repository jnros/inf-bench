import torch

# Force math SDP only — no flash/mem-efficient kernels.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

CONFIGS = [
	("MHA", 8, 8),
	("MQA", 1, 8),
	("GQA", 4, 8),
]

def bench(label, Hkv, Hq, B=1, D=64, T=128,
	  seqlens=[256, 512, 1024, 4096, 8192],
	  dtype=torch.float16, device="cuda"):

	print(label)
	for S in seqlens:
		torch.cuda.reset_peak_memory_stats()

		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)

		k_cache = torch.randn((B, Hkv, S, D), device=device, dtype=dtype)
		v_cache = torch.randn((B, Hkv, S, D), device=device, dtype=dtype)
		kvbytes = (k_cache.numel() + v_cache.numel()) * k_cache.element_size()

		lonely_q = torch.randn((B, Hq, 1, D), device=device, dtype=dtype)

		# expand KV heads for GQA (noop when Hkv==Hq, broadcast handles Hkv==1)
		if Hkv > 1 and Hkv != Hq:
			k = k_cache.repeat_interleave(Hq // Hkv, dim=1)
			v = v_cache.repeat_interleave(Hq // Hkv, dim=1)
		else:
			k = k_cache
			v = v_cache

		# warmup
		torch.matmul(lonely_q, k.transpose(2, 3))
		torch.cuda.synchronize()

		start.record()
		for _ in range(T):
			scores = torch.matmul(lonely_q, k.transpose(2, 3))
			scaled = scores / (D ** 0.5)
			awts = torch.nn.functional.softmax(scaled, dim=-1)
			Z = torch.matmul(awts, v)
		end.record()
		end.synchronize()

		peak_alloc = torch.cuda.max_memory_allocated()
		time_unit = start.elapsed_time(end) / T

		kstr = str(tuple(k_cache.shape))
		print(
			f"  S={S:5d} k={kstr:<20s}"
			f"  KV={kvbytes/1e6:7.2f}MB"
			f"  peak={peak_alloc/1e6:7.2f}MB"
			f"  {time_unit:.3f} ms/tok"
		)
	print()

def main():
	torch.manual_seed(0)
	assert torch.cuda.is_available(), "CUDA required"

	print("GPU:", torch.cuda.get_device_name(0))
	print("torch:", torch.__version__, "cuda:", torch.version.cuda)
	print()

	for label, Hkv, Hq in CONFIGS:
		bench(label, Hkv, Hq)

if __name__ == "__main__":
	main()
