import os
import re
import torch
import csv

# Force math SDP only — no flash/mem-efficient kernels.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

CONFIGS = [
	("MHA", 8, 8),
	("MQA", 1, 8),
	("GQA", 4, 8),
]

SEQLENS_BASE = [256, 512, 1024, 4096, 8192]
SEQLENS_EXT = [256, 512, 1024, 4096, 8192, 16384, 32768, 65536]

def gpu_slug(name):
	s = name.lower().strip()
	s = re.sub(r'[^a-z0-9]+', '-', s)
	return s.strip('-')

def pick_seqlens():
	vram = torch.cuda.get_device_properties(0).total_memory
	if vram >= 24 * 1024**3:
		return SEQLENS_EXT
	return SEQLENS_BASE

def bench(label, Hkv, Hq, B=1, D=64, T=128,
	  seqlens=None, dtype=torch.float16, device="cuda"):

	if seqlens is None:
		seqlens = pick_seqlens()

	elem = torch.finfo(dtype).bits // 8
	rows = []
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

		# bytes touched: read Q + read K + write/read scores +
		# write/read scaled + write/read awts + read V + write Z
		bytes_touched = B * Hq * elem * (2 * D + 2 * S * D + 5 * S)
		bw_gbs = bytes_touched / (time_unit * 1e-3) / 1e9

		kstr = str(tuple(k_cache.shape))
		print(
			f"  S={S:5d} k={kstr:<20s}"
			f"  KV={kvbytes/1e6:7.2f}MB"
			f"  peak={peak_alloc/1e6:7.2f}MB"
			f"  {time_unit:.3f} ms/tok"
		)
		rows.append({
			"label": label,
			"S": S,
			"Hkv": Hkv,
			"Hq": Hq,
			"kv_mb": round(kvbytes / 1e6, 2),
			"peak_mb": round(peak_alloc / 1e6, 2),
			"ms_tok": round(time_unit, 3),
			"bw_gbs": round(bw_gbs, 2),
		})
	print()
	return rows

def plot(results, gpu_name, outpath):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	fig, axes = plt.subplots(1, 3, figsize=(15, 5))

	by_label = {}
	for row in results:
		lbl = row["label"]
		if lbl not in by_label:
			by_label[lbl] = {"S": [], "ms_tok": [],
					 "kv_mb": [], "bw_gbs": []}
		d = by_label[lbl]
		d["S"].append(row["S"])
		d["ms_tok"].append(row["ms_tok"])
		d["kv_mb"].append(row["kv_mb"])
		d["bw_gbs"].append(row["bw_gbs"])

	panels = [
		(0, "ms_tok", "ms / tok"),
		(1, "kv_mb", "KV cache (MB)"),
		(2, "bw_gbs", "bandwidth (GB/s)"),
	]

	for idx, key, ylabel in panels:
		ax = axes[idx]
		for lbl, d in by_label.items():
			ax.plot(d["S"], d[key], "o-", label=lbl)
		ax.set_xscale("log", base=2)
		ax.set_xlabel("sequence length")
		ax.set_ylabel(ylabel)
		ax.legend()
		ax.grid(True, alpha=0.3)

	fig.suptitle(f"{gpu_name} \u2014 decode attention scaling", fontsize=14)
	fig.text(0.5, 0.01, "math SDP only (flash/mem-efficient disabled)",
		 ha="center", fontsize=9, style="italic")
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	fig.savefig(outpath, dpi=150)
	plt.close(fig)
	print(f"saved {outpath}")

def main():
	torch.manual_seed(0)
	assert torch.cuda.is_available(), "CUDA required"

	gpu_name = torch.cuda.get_device_name(0)
	slug = gpu_slug(gpu_name)
	os.makedirs("results", exist_ok=True)

	print("GPU:", gpu_name)
	print("torch:", torch.__version__, "cuda:", torch.version.cuda)
	print()

	results = []
	for label, Hkv, Hq in CONFIGS:
		results.extend(bench(label, Hkv, Hq))

	# inject gpu column
	for r in results:
		r["gpu"] = gpu_name

	fields = ["gpu", "label", "S", "Hkv", "Hq", "kv_mb",
		  "peak_mb", "ms_tok", "bw_gbs"]
	csvpath = f"results/{slug}.csv"
	with open(csvpath, "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fields)
		w.writeheader()
		w.writerows(results)
	print(f"saved {csvpath}")

	plot(results, gpu_name, f"results/{slug}.png")

if __name__ == "__main__":
	main()
