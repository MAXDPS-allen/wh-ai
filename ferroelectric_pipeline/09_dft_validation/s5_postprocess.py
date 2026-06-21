#!/usr/bin/env python3
"""
Stage 5: 后处理 —— Berry 相极化分支跟踪 + 物性提取 + 质量判据
=====================================================================
复刻 Smidt et al. 2020 的后处理 (其作者贡献到 pymatgen.analysis.ferroelectricity):
  1. 解析路径上各结构的电子极化 (Berry 相) 与离子极化
  2. 分支跟踪: 把各像的极化校正到同一极化分支 (避开极化量子跳变)
  3. 自发极化 Ps = 极性端点 − 非极性端点 (同一分支)
  4. 能量路径 → 双势阱深度 / 切换能垒
  5. 金属性检查 (路径最小带隙)
  6. 质量判据: 极化/能量路径平滑度 (样条偏差)
  7. 综合判定该候选是否为 (优质) 铁电体

两种输入模式:
  - dirs : 从 Stage 4 的 VASP 运行目录解析 (生产)
  - arrays: 从预先算好的极化/能量数组 (用于对拍验证, 如 workflow_data.json)

依赖: pymatgen.analysis.ferroelectricity.polarization
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config

try:
    from pymatgen.analysis.ferroelectricity.polarization import (
        Polarization, EnergyTrend)
    from pymatgen.core import Structure
    _HAS_PMG = True
except Exception:
    _HAS_PMG = False


# ---------------------------------------------------------------------------
# 极化分支跟踪 + 物性 (核心)
# ---------------------------------------------------------------------------
def analyze_polarization(p_elecs, p_ions, structures):
    """用 pymatgen 做同分支极化跟踪, 返回自发极化 (μC/cm²) 与路径。"""
    pol = Polarization(p_elecs, p_ions, structures)
    same_branch = pol.get_same_branch_polarization_data(
        convert_to_muC_per_cm2=True, all_in_polar=True)
    Ps_vec = pol.get_polarization_change()           # 极性−非极性 (μC/cm²)
    Ps_norm = pol.get_polarization_change_norm()
    smoothness = pol.smoothness(convert_to_muC_per_cm2=True)  # 各分量样条偏差
    return {
        "same_branch_polarization": np.asarray(same_branch).tolist(),
        "Ps_vector": np.asarray(Ps_vec).ravel().tolist(),
        "Ps_norm": float(Ps_norm),
        "polarization_smoothness": [float(s) for s in np.atleast_1d(smoothness)],
    }


def analyze_energy(energies_per_atom):
    """能量路径 → 双势阱深度与能垒 (相对极性端点, meV/atom)。"""
    e = np.asarray(energies_per_atom, dtype=float)
    e_rel = (e - e[-1]) * 1000.0
    et = None
    try:
        et = EnergyTrend(list(e))
        e_smooth = et.smoothness()
        max_jump = et.max_spline_jump()
    except Exception:
        e_smooth, max_jump = float("nan"), float("nan")
    return {
        "dw_depth_meV": float(e_rel[0]),
        "path_barrier_meV": float(e_rel.max()),
        "energy_smoothness": float(e_smooth),
        "energy_max_spline_jump": float(max_jump),
    }


def verdict(Ps_norm, gap_min, gap_polar, pol_smooth, e_smooth, dw_depth):
    """综合判定 (复刻论文质量判据 + 物理必要条件)。"""
    checks = {
        "insulating": gap_min >= config.METAL_GAP_EV,          # 非金属 (必要)
        "polar_endpoint_gapped": gap_polar >= config.METAL_GAP_EV,
        "pol_path_smooth": (max(pol_smooth) if pol_smooth else 9e9) < config.POL_SMOOTH_TOL,
        "energy_path_smooth": (e_smooth if e_smooth == e_smooth else 9e9) < config.ENERGY_SMOOTH_TOL,
        "nonzero_polarization": Ps_norm > 0.1,
    }
    is_ferroelectric = checks["insulating"] and checks["nonzero_polarization"]
    is_high_quality = all(checks.values())
    # 优质铁电的附加物性偏好 (用户关切: 不只是极性, 而是"好"的铁电)
    quality_flags = {
        "polar_ground_state": dw_depth > 0,        # 极性相为基态
        "large_polarization": Ps_norm >= 5.0,      # μC/cm² (高于数据集中位数)
    }
    return {
        "checks": checks,
        "is_ferroelectric": bool(is_ferroelectric),
        "is_high_quality": bool(is_high_quality),
        "quality_flags": quality_flags,
    }


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------
def parse_from_dirs(vasp_root: Path):
    from pymatgen.io.vasp.outputs import Outcar, Vasprun
    images = sorted((vasp_root).glob("polar_image_*"))
    p_elecs, p_ions, structures, energies, gaps = [], [], [], [], []
    for d in images:
        oc = Outcar(str(d / "OUTCAR"))
        p_elecs.append(oc.p_elec)
        p_ions.append(oc.p_ion)
        # 能量/带隙取自对应 static
        sd = vasp_root / d.name.replace("polar_", "static_")
        vr = Vasprun(str(sd / "vasprun.xml"), parse_dos=False, parse_eigen=True)
        structures.append(vr.final_structure)
        energies.append(vr.final_energy / len(vr.final_structure))
        gap, _, _, _ = vr.eigenvalue_band_properties
        gaps.append(gap)
    return p_elecs, p_ions, structures, energies, gaps


def run_from_dirs(cid_dir: Path):
    vasp_root = cid_dir / "vasp"
    p_elecs, p_ions, structures, energies, gaps = parse_from_dirs(vasp_root)
    return assemble(cid_dir, p_elecs, p_ions, structures, energies, gaps)


def assemble(cid_dir: Path, p_elecs, p_ions, structures, energies, gaps):
    pol = analyze_polarization(p_elecs, p_ions, structures)
    ene = analyze_energy(energies)
    gap_min = float(np.min(gaps)); gap_polar = float(gaps[-1])
    vd = verdict(pol["Ps_norm"], gap_min, gap_polar,
                 pol["polarization_smoothness"], ene["energy_smoothness"], ene["dw_depth_meV"])
    result = {
        "cid": cid_dir.name,
        "Ps_norm_uC_cm2": pol["Ps_norm"],
        "Ps_vector": pol["Ps_vector"],
        "gap_min_eV": gap_min,
        "gap_polar_eV": gap_polar,
        "bandgaps": [float(g) for g in gaps],
        **ene,
        "polarization_smoothness": pol["polarization_smoothness"],
        "same_branch_polarization": pol["same_branch_polarization"],
        **vd,
    }
    json.dump(result, open(cid_dir / "dft_validation_result.json", "w"), indent=2)
    print(f"[{cid_dir.name}] Ps={pol['Ps_norm']:.2f} μC/cm² | gap_min={gap_min:.2f} eV | "
          f"barrier={ene['path_barrier_meV']:.1f} meV/atom | "
          f"ferroelectric={vd['is_ferroelectric']} high_quality={vd['is_high_quality']}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Post-process DFT outputs -> Ps/barrier/gap + verdict")
    ap.add_argument("cid_dir", type=Path)
    args = ap.parse_args()
    if not _HAS_PMG:
        raise SystemExit("需要 pymatgen: conda activate fe_dft")
    run_from_dirs(args.cid_dir)


if __name__ == "__main__":
    main()
