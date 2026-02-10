import csv
import glob
import os
import sys

# (match_substr, display_name, peak_bw_gbs)
# Matched against gpu column (case-insensitive), most specific first.
# Sources: results/sources.txt
GPUS = [
	("h100 80gb hbm3",  "H100 SXM",   3350),
	("h100 pcie",       "H100 PCIe",   2039),
	("a100-sxm",        "A100 SXM4",   2039),
	("rtx 2060",        "RTX 2060",     336),
]

def gpu_lookup(gpu_name):
	"""Return (display_name, peak_bw, sort_idx) or fallback."""
	low = gpu_name.lower()
	for i, (substr, name, bw) in enumerate(GPUS):
		if substr in low:
			return name, bw, i
	return gpu_name, None, len(GPUS)

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

	ordered = sorted(mha.keys(), key=lambda g: gpu_lookup(g)[2])
	for ci, gpu in enumerate(ordered):
		rows = mha[gpu]
		xs = sorted(int(r["S"]) for r in rows)
		by_s = {int(r["S"]): r for r in rows}
		color = colors[ci % len(colors)]
		label, peak, _ = gpu_lookup(gpu)

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

if __name__ == "__main__":
	gpus = load_csvs()
	print(f"loaded {len(gpus)} GPU(s): {', '.join(sorted(gpus))}")
	plot(gpus)
