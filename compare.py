import csv
import glob
import os
import sys

# Theoretical peak memory bandwidth (GB/s) per GPU.
# Matched against gpu column (case-insensitive), most specific first.
# Sources: results/sources.txt
PEAK_BW = [
	("h100 80gb hbm3",  3350),	# SXM5 80GB HBM3
	("h100 pcie",       2039),	# PCIe 80GB HBM2e
	("h100 nvl",        3938),	# NVL 94GB HBM3
	("a100-sxm",        2039),	# SXM4 80GB HBM2e
	("a100-pcie-80",    1935),	# PCIe 80GB HBM2e
	("a100-pcie-40",    1555),	# PCIe 40GB HBM2
	("a100",            2039),	# SXM fallback
	("a10g",             600),	# 24GB GDDR6 384-bit
	("rtx 2060",         336),	# 192-bit @ 14 Gbps
	("rtx 3060",         360),	# 192-bit @ 15 Gbps
	("rtx 3070",         448),	# 256-bit @ 14 Gbps
	("rtx 3080",         760),	# 320-bit @ 19 Gbps GDDR6X
	("rtx 3090",         936),	# 384-bit @ 19.5G GDDR6X
	("rtx 4070",         504),	# 192-bit @ 21 Gbps GDDR6X
	("rtx 4080",         717),	# 256-bit @ 22.4G GDDR6X
	("rtx 4090",        1008),	# 384-bit @ 21 Gbps GDDR6X
	("l40s",             864),	# 384-bit GDDR6 ECC
	("l40",              864),	# 384-bit GDDR6 ECC
	("l4",               300),	# 192-bit GDDR6
]

# Display names and legend order (top to bottom).
# Matched against gpu column same as PEAK_BW.
GPU_ORDER = [
	("h100 80gb hbm3",	"H100 SXM"),
	("h100 nvl",		"H100 NVL"),
	("h100 pcie",		"H100 PCIe"),
	("a100-sxm",		"A100 SXM4"),
	("a100-pcie-80",	"A100 PCIe 80GB"),
	("a100-pcie-40",	"A100 PCIe 40GB"),
	("a100",		"A100"),
	("a10g",		"A10G"),
	("rtx 4090",		"RTX 4090"),
	("rtx 4080",		"RTX 4080"),
	("rtx 4070",		"RTX 4070"),
	("rtx 3090",		"RTX 3090"),
	("rtx 3080",		"RTX 3080"),
	("rtx 3070",		"RTX 3070"),
	("rtx 3060",		"RTX 3060"),
	("rtx 2060",		"RTX 2060"),
	("l40s",		"L40S"),
	("l40",			"L40"),
	("l4",			"L4"),
]

def match_peak(gpu_name):
	low = gpu_name.lower()
	for substr, bw in PEAK_BW:
		if substr in low:
			return bw
	return None

def display_name(gpu_name):
	low = gpu_name.lower()
	for substr, name in GPU_ORDER:
		if substr in low:
			return name
	return gpu_name

def gpu_sort_key(gpu_name):
	low = gpu_name.lower()
	for i, (substr, _) in enumerate(GPU_ORDER):
		if substr in low:
			return i
	return len(GPU_ORDER)

def load_csvs(pattern="results/*.csv"):
	files = sorted(glob.glob(pattern))
	if not files:
		print("no CSV files found in results/", file=sys.stderr)
		sys.exit(1)

	gpus = {}  # gpu_name -> list of rows
	for path in files:
		with open(path, newline="") as f:
			rows = list(csv.DictReader(f))
		if not rows:
			continue
		name = rows[0].get("gpu", os.path.basename(path))
		gpus[name] = rows
	return gpus

def plot(gpus, outpath="results/compare.png"):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	# Filter to MHA only
	mha = {}
	for gpu, rows in gpus.items():
		mha[gpu] = [r for r in rows if r["label"] == "MHA"]

	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

	ordered = sorted(mha.keys(), key=gpu_sort_key)
	for ci, gpu in enumerate(ordered):
		rows = mha[gpu]
		xs = sorted(int(r["S"]) for r in rows)
		by_s = {int(r["S"]): r for r in rows}
		color = colors[ci % len(colors)]
		peak = match_peak(gpu)
		label = display_name(gpu)

		ms = [float(by_s[s]["ms_tok"]) for s in xs]
		bw = [float(by_s[s]["bw_gbs"]) for s in xs]

		# panel 0: ms/tok
		axes[0].plot(xs, ms, "o-", label=label, color=color)

		# panel 1: absolute bandwidth
		axes[1].plot(xs, bw, "o-", label=label, color=color)
		if peak:
			axes[1].axhline(peak, linestyle="--", color=color,
					alpha=0.5, linewidth=1,
					label=f"peak ({peak})")

		# panel 2: % of peak bandwidth
		if peak:
			pct = [b / peak * 100 for b in bw]
			axes[2].plot(xs, pct, "o-", label=label, color=color)

	# panel 2: 100% reference line
	axes[2].axhline(100, linestyle="--", color="grey",
			alpha=0.5, linewidth=1)

	labels = [
		(0, "ms / tok"),
		(1, "bandwidth (GB/s)"),
		(2, "% of peak bandwidth"),
	]
	for idx, ylabel in labels:
		ax = axes[idx]
		ax.set_xscale("log", base=2)
		ax.set_xlabel("sequence length")
		ax.set_ylabel(ylabel)
		ax.legend(fontsize=7)
		ax.grid(True, alpha=0.3)

	fig.suptitle("cross-GPU decode attention (MHA)", fontsize=14)
	fig.text(0.5, 0.01, "math SDP only (flash/mem-efficient disabled)",
		 ha="center", fontsize=9, style="italic")
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	fig.savefig(outpath, dpi=150)
	plt.close(fig)
	print(f"saved {outpath}")

def plot_pct(gpus, outpath="results/compare-pct.png"):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	mha = {}
	for gpu, rows in gpus.items():
		mha[gpu] = [r for r in rows if r["label"] == "MHA"]

	fig, ax = plt.subplots(figsize=(8, 5))
	colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

	ordered = sorted(mha.keys(), key=gpu_sort_key)
	for ci, gpu in enumerate(ordered):
		rows = mha[gpu]
		xs = sorted(int(r["S"]) for r in rows)
		by_s = {int(r["S"]): r for r in rows}
		color = colors[ci % len(colors)]
		peak = match_peak(gpu)
		if not peak:
			continue
		bw = [float(by_s[s]["bw_gbs"]) for s in xs]
		pct = [b / peak * 100 for b in bw]
		ax.plot(xs, pct, "o-", label=display_name(gpu), color=color)

	ax.axhline(100, linestyle="--", color="grey",
		   alpha=0.5, linewidth=1)
	ax.set_xscale("log", base=2)
	ax.set_xlabel("sequence length")
	ax.set_ylabel("% of peak bandwidth")
	ax.legend(fontsize=8)
	ax.grid(True, alpha=0.3)
	ax.set_title("cross-GPU decode attention (MHA)", fontsize=14)
	fig.text(0.5, 0.01, "math SDP only (flash/mem-efficient disabled)",
		 ha="center", fontsize=9, style="italic")
	plt.tight_layout(rect=[0, 0.03, 1, 0.97])
	fig.savefig(outpath, dpi=150)
	plt.close(fig)
	print(f"saved {outpath}")

if __name__ == "__main__":
	gpus = load_csvs()
	print(f"loaded {len(gpus)} GPU(s): {', '.join(sorted(gpus))}")
	plot(gpus)
	plot_pct(gpus)
