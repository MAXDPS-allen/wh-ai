#!/usr/bin/env python3
"""
对标 Ricci 2024 (npj) 的说服力图:
  图8 (铁电landscape, 对应其 Ps-能垒图): 636 个已发表铁电 (Ricci+Smidt) 的 Ps-能量分布,
       标注我们独立验证的 SrAlGeH/GeTe + 说明我们的非极性互补搜索空间。
  图9 (ML vs DFT parity): 我们的 Ps 模型 (15 ms) 在留出集上重现 DFT 级 Ps (~小时)。
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
prior = json.load(open(HERE / "prior_landscape.json"))
ps = np.array([r["Ps"] for r in prior])
en = np.array([r["energy"] if r["energy"] is not None else np.nan for r in prior])
proj = np.array([r["project"] for r in prior])
def find(frag):
    for r in prior:
        if frag.lower() in r["formula"].lower():
            return r
    return None
sral = find("SrAlGe"); gete = find("GeTe")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Our work in the context of published high-throughput ferroelectrics "
             "(Ricci 2024 + Smidt 2020, n=636)", fontsize=12, fontweight="bold")

# 图8a: Ps-energy landscape
a = ax[0]
m_ext = proj == "ferroelectrics_ext"; m_sm = proj == "ferroelectrics"
a.scatter(en[m_sm], ps[m_sm], s=16, alpha=0.45, c="#7f7f7f", label="Smidt 2020 (250)")
a.scatter(en[m_ext], ps[m_ext], s=16, alpha=0.5, c="#1f77b4", label="Ricci 2024 (386)")
for r, name, col in [(sral, "SrAlGeH", "#d62728"), (gete, "GeTe", "#2ca02c")]:
    if r and r["energy"] is not None:
        a.scatter([r["energy"]], [r["Ps"]], s=160, marker="*", c=col, edgecolor="k",
                  zorder=5, label=f"{name} (we validated)")
a.set_xlim(0, 300); a.set_ylim(0, 160)
a.set_xlabel("energy difference polar↔nonpolar (meV/atom)")
a.set_ylabel("spontaneous polarization Pₛ (μC/cm²)")
a.set_title("(a) Ferroelectric landscape — we reproduce & validate within it")
a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.3)

# 图9: ML vs DFT parity (held-out)
b = ax[1]
pf = HERE / "ps_parity.json"
if pf.exists():
    d = json.load(open(pf))
    t = np.array(d["true_Ps"]); p = np.array(d["pred_Ps"]); sc = np.array(d["source"])
    from scipy.stats import spearmanr
    rho = spearmanr(t, p).correlation
    mae = np.abs(t - p).mean()
    for s, col, lab in [("smidt", "#7f7f7f", "Smidt"), ("ricci", "#1f77b4", "Ricci")]:
        mm = sc == s
        b.scatter(t[mm], p[mm], s=22, alpha=0.6, c=col, label=lab)
    lim = [0, max(t.max(), p.max()) * 1.05]
    b.plot(lim, lim, "--", c="k", lw=1, label="ideal")
    b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel("DFT spontaneous polarization Pₛ (μC/cm², ~3 h/material)")
    b.set_ylabel("ML-predicted Pₛ (μC/cm², ~15 ms/material)")
    b.set_title(f"(b) ML reproduces DFT-level Pₛ on held-out set\nSpearman ρ={rho:.2f}, MAE={mae:.1f} μC/cm² — ~8×10⁵× faster")
    b.legend(fontsize=8); b.grid(alpha=0.3)
else:
    b.text(0.5, 0.5, "ps_parity.json pending", ha="center")

fig.tight_layout(rect=[0, 0, 1, 0.93])
out = HERE / "report_fig8_paper_comparison.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out)
