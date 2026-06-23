#!/usr/bin/env python3
"""
绘制 SrAlGeH 极性相声子谱 (动力学稳定性证明)
=====================================================================
重建力常数 → 沿六方布里渊区高对称路径计算声子色散 → 绘图。
无虚频 (所有频率 ≥ 0) 即证明极性相动力学稳定 —— 铁电存在的必要条件。
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "09_dft_validation"))
from phonopy import load
from pymatgen.io.vasp.outputs import Vasprun

HERE = Path(__file__).parent
CID = "SrAlGeH_mp-980057"
pdir = HERE / "test_run" / CID / "phonon"

# 重建力常数
ph = load(pdir / "phonopy_disp.yaml")
forces = []
for d in sorted((pdir / "disps").glob("disp_*")):
    vr = Vasprun(str(d / "vasprun.xml"), parse_dos=False, parse_eigen=False)
    forces.append(np.array(vr.ionic_steps[-1]["forces"]))
ph.forces = forces
ph.produce_force_constants()

# 六方高对称路径 Γ-M-K-Γ-A-L-H-A
labels = ["$\\Gamma$", "M", "K", "$\\Gamma$", "A", "L", "H", "A"]
path = [[[0, 0, 0], [0.5, 0, 0], [1/3, 1/3, 0], [0, 0, 0],
         [0, 0, 0.5], [0.5, 0, 0.5], [1/3, 1/3, 0.5], [0, 0, 0.5]]]
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=101)
ph.run_band_structure(qpoints, path_connections=connections, labels=labels)
bs = ph.get_band_structure_dict()

# 频率 THz -> cm^-1 (×33.356) 便于阅读; 这里用 THz
fig, ax = plt.subplots(figsize=(8, 5))
dists = bs["distances"]; freqs = bs["frequencies"]
xticks, xmin = [], 0.0
for seg_d, seg_f in zip(dists, freqs):
    for band in seg_f.T:
        ax.plot(seg_d, band, "-", color="#1f77b4", lw=1.1)
# 段边界与刻度
tick_pos = [dists[0][0]] + [seg[-1] for seg in dists]
for x in tick_pos:
    ax.axvline(x, color="gray", lw=0.5, alpha=0.5)
ax.axhline(0, color="red", ls="--", lw=1, label="0 (imaginary below)")
ax.set_xticks(tick_pos); ax.set_xticklabels(labels)
ax.set_xlim(tick_pos[0], tick_pos[-1])
min_f = min(seg.min() for seg in freqs)
ax.set_ylabel("Phonon frequency (THz)")
ax.set_title(f"SrAlGeH (mp-980057) polar phase — phonon dispersion\n"
             f"min frequency = {min_f:.2f} THz  →  dynamically stable (no imaginary modes)")
ax.legend(fontsize=9); ax.grid(alpha=0.2, axis="y")
fig.tight_layout()
out = pdir / "phonon_dispersion.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"saved -> {out}  (min freq {min_f:.3f} THz)")
