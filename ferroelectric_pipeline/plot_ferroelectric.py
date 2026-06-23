#!/usr/bin/env python3
"""
绘制反映铁电性的决定性第一性原理图
=====================================================================
参考铁电文献 (Smidt et al. Sci.Data 2020; Landau 理论 / Cochran 软模; Resta &
Vanderbilt 现代极化理论), 铁电性的决定性判据图为:

  (A) 能量双势阱  E vs 畸变 λ      —— 极性相为基态、经中心对称相切换 (±P 简并)
  (B) 极化 vs 畸变 P vs λ           —— 序参量随畸变发展 (软模失稳)
  (C) Landau 双势阱 E vs P          —— 铁电的标志性自由能曲线
  (D) 带隙 vs 畸变 E_g vs λ         —— 绝缘必要条件 + 带隙-极化耦合

数据来自 09_dft_validation 的 DFT 计算 (能量、带隙) 与点电荷极化估计。
λ=0 为非极性 (高对称) 端点, λ=1 为极性端点; 利用 ±P 对称性镜像为 [-1,1] 双势阱。

用法: conda activate fe_dft && python plot_ferroelectric.py
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "09_dft_validation"))
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.analysis.bond_valence import BVAnalyzer

HERE = Path(__file__).parent
RUN = HERE / "test_run"
E_TO_UCCM2 = 1602.176634     # e/Å² -> μC/cm²
METAL_GAP = 0.01

CANDS = [
    ("GeTe_mp-938", "GeTe (mp-938)", "#d62728"),
    ("CaAlSiH_mp-568177", "CaAlSiH (mp-568177)", "#1f77b4"),
    ("SrAlGeH_mp-980057", "SrAlGeH (mp-980057)", "#2ca02c"),
]


def extract(cid):
    """返回沿路径 (λ=0 非极性 → λ=1 极性) 的 能量(meV/atom,相对极性)、带隙、点电荷极化。"""
    vdir = RUN / cid / "vasp"
    E, G = [], []
    for i in range(10):
        vr = Vasprun(str(vdir / f"static_image_{i:02d}" / "vasprun.xml"),
                     parse_dos=False, parse_eigen=True)
        E.append(vr.final_energy / len(vr.final_structure))
        G.append(vr.eigenvalue_band_properties[0])
    E = np.array(E); G = np.array(G)
    e_rel = (E - E[-1]) * 1000.0           # meV/atom, 相对极性相

    # 点电荷极化沿路径 (相对非极性), 用名义氧化态
    polar = Structure.from_dict(json.load(open(RUN / cid / "polar.json")))
    nonpolar = Structure.from_dict(json.load(open(RUN / cid / "nonpolar.json")))
    ox = None
    try:
        ox = BVAnalyzer().get_valences(polar)
        if all(abs(o) < 1e-6 for o in ox):
            ox = None
    except Exception:
        ox = None
    if ox is None:                       # 回退: 组分氧化态猜测 (适用于 Zintl 相)
        try:
            guess = polar.composition.oxi_state_guesses()
            if guess:
                ox = [guess[0][s.specie.symbol] for s in polar]
        except Exception:
            ox = None
    if ox is None:                       # 二次回退: 名义氧化态映射 (本测试体系)
        nominal = {"Sr": 2, "Ca": 2, "Al": 3, "Si": -4, "Ge": -4, "H": -1,
                   "Te": -2}
        try:
            ox = [nominal[s.specie.symbol] for s in polar]
        except Exception:
            ox = None
    P = None
    if ox is not None:
        # 总位移用最小镜像计算一次 (避免逐 λ 重复折叠导致的极化分支跳变伪影);
        # 线性插值下 P(λ) = λ · P_total (点电荷模型严格线性发展)。
        V = polar.lattice.volume
        dP_tot = np.zeros(3)
        for k in range(len(polar)):
            df = polar[k].frac_coords - nonpolar[k].frac_coords
            df -= np.round(df)
            dP_tot += ox[k] * polar.lattice.get_cartesian_coords(df)
        Ps_tot = np.linalg.norm(dP_tot) / V * E_TO_UCCM2
        P = np.linspace(0, 1, 10) * Ps_tot
    return e_rel, G, P


def mirror(lam, y, even=True):
    """利用 ±P 对称性把 λ∈[0,1] 镜像为 [-1,1]。even: 能量/带隙偶对称; 极化奇对称。"""
    lam_full = np.concatenate([-lam[::-1], lam[1:]])
    y_full = np.concatenate([y[::-1], y[1:]]) if even else np.concatenate([-y[::-1], y[1:]])
    return lam_full, y_full


def main():
    lam = np.linspace(0, 1, 10)
    data = {}
    for cid, label, color in CANDS:
        try:
            data[cid] = extract(cid)
        except Exception as e:
            print(f"skip {cid}: {e}")

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Decisive first-principles signatures of ferroelectricity\n"
                 "(distortion path: λ=0 nonpolar/high-symmetry → λ=1 polar)",
                 fontsize=13, fontweight="bold")

    # (A) 能量双势阱 E vs λ (镜像)
    a = ax[0, 0]
    for cid, label, color in CANDS:
        if cid not in data: continue
        e_rel, G, P = data[cid]
        lf, ef = mirror(lam, e_rel, even=True)
        a.plot(lf, ef, "o-", color=color, label=label, ms=4)
    a.axvline(0, ls=":", c="gray", lw=1)
    a.set_xlabel("distortion coordinate  λ  (±polar variant)")
    a.set_ylabel("Energy  (meV/atom, rel. polar)")
    a.set_title("(A) Energy double-well  —  polar phase is the ground state")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # (B) 极化 vs λ (点电荷, 镜像奇对称)
    b = ax[0, 1]
    for cid, label, color in CANDS:
        if cid not in data: continue
        e_rel, G, P = data[cid]
        if P is None:
            continue
        lf, pf = mirror(lam, P, even=False)
        b.plot(lf, pf, "s-", color=color, label=label, ms=4)
    b.axhline(0, ls=":", c="gray", lw=1); b.axvline(0, ls=":", c="gray", lw=1)
    b.set_xlabel("distortion coordinate  λ")
    b.set_ylabel("Polarization  (μC/cm², point-charge)")
    b.set_title("(B) Polarization vs distortion  —  order parameter (soft mode)")
    b.legend(fontsize=8); b.grid(alpha=0.3)

    # (C) Landau 双势阱 E vs P
    c = ax[1, 0]
    for cid, label, color in CANDS:
        if cid not in data: continue
        e_rel, G, P = data[cid]
        if P is None:
            continue
        # 以 ±P 镜像画 W 形
        Pf = np.concatenate([-P[::-1], P[1:]])
        Ef = np.concatenate([e_rel[::-1], e_rel[1:]])
        c.plot(Pf, Ef, "o-", color=color, label=label, ms=4)
    c.axvline(0, ls=":", c="gray", lw=1)
    c.set_xlabel("Polarization  P  (μC/cm²)")
    c.set_ylabel("Energy  (meV/atom, rel. polar)")
    c.set_title("(C) Landau double-well  E(P)  —  hallmark of ferroelectricity")
    c.legend(fontsize=8); c.grid(alpha=0.3)

    # (D) 带隙 vs λ
    d = ax[1, 1]
    for cid, label, color in CANDS:
        if cid not in data: continue
        e_rel, G, P = data[cid]
        d.plot(lam, G, "^-", color=color, label=label, ms=5)
    d.axhline(METAL_GAP, ls="--", c="red", lw=1, label="metal threshold (10 meV)")
    d.set_xlabel("distortion coordinate  λ  (0=nonpolar → 1=polar)")
    d.set_ylabel("Band gap  (eV)")
    d.set_title("(D) Band gap evolution  —  insulating requirement & gap–P coupling")
    d.legend(fontsize=8); d.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = HERE / "test_run" / "ferroelectric_signatures.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")

    # 另存每个候选的单图 (能量+带隙双轴), 便于细看
    for cid, label, color in CANDS:
        if cid not in data: continue
        e_rel, G, P = data[cid]
        fig2, ax1 = plt.subplots(figsize=(6.5, 4.5))
        lf, ef = mirror(lam, e_rel, even=True)
        ax1.plot(lf, ef, "o-", color="#d62728", label="Energy")
        ax1.set_xlabel("distortion λ (±polar)"); ax1.set_ylabel("Energy (meV/atom)", color="#d62728")
        ax1.axvline(0, ls=":", c="gray"); ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        lf2, gf = mirror(lam, G, even=True)
        ax2.plot(lf2, gf, "^-", color="#1f77b4", label="Band gap")
        ax2.set_ylabel("Band gap (eV)", color="#1f77b4")
        dwell = e_rel[0]
        ax1.set_title(f"{label}\ndouble-well depth = {dwell:.1f} meV/atom, "
                      f"gap {G[0]:.2f}→{G[-1]:.2f} eV")
        fig2.tight_layout()
        o2 = HERE / "test_run" / f"signature_{cid}.png"
        fig2.savefig(o2, dpi=150, bbox_inches="tight")
        print(f"saved -> {o2}")


if __name__ == "__main__":
    main()
