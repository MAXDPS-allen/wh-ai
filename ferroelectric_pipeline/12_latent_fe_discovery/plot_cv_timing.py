#!/usr/bin/env python3
"""图3 (报告补充): 5-折 CV 稳定性 + ML/DFT 计时对比。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
cv = json.load(open(HERE / "cv_results.json"))
tm = json.load(open(HERE / "timing_results.json"))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
fig.suptitle("Model stability (5-fold CV) and ML-vs-DFT throughput", fontsize=13, fontweight="bold")

# (a) CV error bars
metrics = [("Tier-1\nAUC", cv["tier1_AUC"]),
           ("Tier-2\nsign-AUC", cv["tier2_sign_AUC"]),
           ("Tier-2\nR²", cv["tier2_R2"])]
labels = [m[0] for m in metrics]
means = [m[1][0] for m in metrics]; stds = [m[1][1] for m in metrics]
bars = ax[0].bar(labels, means, yerr=stds, capsize=8,
                 color=["#1f77b4", "#2ca02c", "#2ca02c"], alpha=0.85)
ax[0].axhline(0.5, ls="--", color="gray", lw=1, label="AUC random")
ax[0].set_ylim(0, 1.0); ax[0].set_ylabel("score")
ax[0].set_title(f"(a) 5-fold cross-validation (mean ± std, N={1534})\n"
                f"Tier-1 enrichment {cv['tier1_enrichment'][0]:.1f}±{cv['tier1_enrichment'][1]:.1f}× ; "
                "equivariance err ~1e-4")
for b, m, s in zip(bars, means, stds):
    ax[0].annotate(f"{m:.2f}±{s:.2f}", (b.get_x()+b.get_width()/2, m+s+0.02), ha="center", fontsize=9)
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")

# (b) timing comparison (log scale)
ml_ms = tm["ml_inference"]["forward_per_material_ms"]
dft = tm["dft_walltimes"]
per_mat_hr = np.mean([d["total_wall_hr"] for d in dft.values()])
labels2 = ["ML inference\n(per material)", "DFT validation\n(per material)"]
secs = [ml_ms / 1e3, per_mat_hr * 3600]
bars2 = ax[1].barh(labels2, secs, color=["#1f77b4", "#d62728"])
ax[1].set_xscale("log")
ax[1].set_xlabel("wall time per material (seconds, log scale)")
speedup = secs[1] / secs[0]
ax[1].set_title(f"(b) Throughput: ML triage vs DFT confirmation\n"
                f"ML {ml_ms:.0f} ms  vs  DFT {per_mat_hr:.1f} h  →  ~{speedup:.0e}× faster")
ax[1].annotate(f"{ml_ms:.0f} ms", (secs[0], 0), va="center", ha="left", fontsize=9, xytext=(5,0), textcoords="offset points")
ax[1].annotate(f"{per_mat_hr:.1f} h", (secs[1], 1), va="center", ha="right", fontsize=9, xytext=(-5,0), textcoords="offset points", color="white")
ax[1].grid(alpha=0.3, axis="x")

fig.tight_layout(rect=[0, 0, 1, 0.92])
out = HERE / "report_fig6_stability_timing.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out)
