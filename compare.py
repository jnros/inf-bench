import csv
import glob
import os
import sys

# Theoretical peak memory bandwidth (GB/s) for known GPUs.
# Substring-matched against gpu column, case-insensitive.
PEAK_BW = {
	"rtx 2060":    336,
	"rtx 3060":    360,
	"rtx 3070":    448,
	"rtx 3080":    760,
	"rtx 3090":    936,
	"rtx 4070":    504,
	"rtx 4080":    717,
	"rtx 4090":   1008,
	"a100":       2039,
	"a10g":        600,
	"h100":       3350,
	"l4":          300,
	"l40":         864,
	"l40s":        864,
}

def match_peak(gpu_name):
	low = gpu_name.lower()
	for substr, bw in PEAK_BW.items():
		if substr in low:
			return bw
	return None

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

	# Shared seqlens (intersection)
	seqsets = [set(int(r["S"]) for r in rows) for rows in mha.values()]
	shared = sorted(set.intersection(*seqsets))
	if not shared:
		print("no shared sequence lengths", file=sys.stderr)
		sys.exit(1)

	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	panels = [
		(0, "ms_tok", "ms / tok"),
		(1, "kv_mb", "KV cache (MB)"),
		(2, "bw_gbs", "bandwidth (GB/s)"),
	]

	colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

	for ci, (gpu, rows) in enumerate(sorted(mha.items())):
		by_s = {int(r["S"]): r for r in rows}
		xs = shared
		color = colors[ci % len(colors)]

		for idx, key, ylabel in panels:
			ax = axes[idx]
			ys = [float(by_s[s][key]) for s in xs]
			ax.plot(xs, ys, "o-", label=gpu, color=color)

		# theoretical peak bandwidth line
		peak = match_peak(gpu)
		if peak:
			axes[2].axhline(peak, linestyle="--", color=color,
					alpha=0.5, linewidth=1,
					label=f"{gpu} peak ({peak})")

	for idx, key, ylabel in panels:
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
