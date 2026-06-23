#!/usr/bin/env python3
"""
Stage 6: 极性相声子(动力学)稳定性
=====================================================================
铁电判定的关键补充: 极性相必须**动力学稳定 (无虚频)**, 否则它不是真实可存在
的相。同时, 高对称非极性相应在 Γ 出现**极性软模虚频** —— 这正是位移型铁电的
微观起源 (Cochran 软模理论)。

方法 (phonopy 有限位移 + VASP 力):
  1. 由弛豫极性结构生成超胞 + 对称独立位移
  2. VASP 高精度静态计算每个位移的原子力 (GPU)
  3. phonopy 组装力常数 → 声子谱; 检查是否有虚频 (动力学稳定性)

输出: {cid}/phonon/  (phonopy.yaml, FORCE_SETS, band.yaml, 稳定性判定)

用法:
  python s6_phonon.py runs/<cid> --supercell 2 2 2          # 生成位移
  python s4_run_cluster.py runs/<cid>/phonon/disps --pattern 'disp_*'   # 跑力
  python s6_phonon.py runs/<cid> --collect                  # 组装 + 判定
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config


def _phonopy_atoms(structure):
    from phonopy.structure.atoms import PhonopyAtoms
    return PhonopyAtoms(symbols=[s.specie.symbol for s in structure],
                        scaled_positions=structure.frac_coords,
                        cell=structure.lattice.matrix)


def generate(cid_dir: Path, supercell=(2, 2, 2), disp=0.01):
    """生成超胞位移结构 + 各自的 VASP 静态输入。"""
    from phonopy import Phonopy
    from pymatgen.core import Structure
    from pymatgen.io.phonopy import get_pmg_structure

    polar = Structure.from_dict(json.load(open(cid_dir / "polar.json")))
    ph = Phonopy(_phonopy_atoms(polar), supercell_matrix=np.diag(supercell))
    ph.generate_displacements(distance=disp)
    supercells = ph.supercells_with_displacements
    pdir = cid_dir / "phonon"
    pdir.mkdir(exist_ok=True)
    ph.save(pdir / "phonopy_disp.yaml")          # 含位移数据集

    disps_root = pdir / "disps"
    disps_root.mkdir(exist_ok=True)
    import s3_make_vasp_inputs as s3
    s3._configure_potcar()                        # 设置 pymatgen POTCAR 路径
    for i, sc in enumerate(supercells):
        st = get_pmg_structure(sc)
        dest = disps_root / f"disp_{i:03d}"
        s3.write_phonon_force(st, dest)          # 粗 k 网格 + 紧 EDIFF, 力计算
    n = len(supercells)
    print(f"[{cid_dir.name}] 声子: {supercell} 超胞 ({len(supercells[0])} 原子), "
          f"{n} 个位移 -> {disps_root}")
    print(f"  下一步: python s4_run_cluster.py {disps_root} --pattern 'disp_*'")
    return n


def collect(cid_dir: Path, supercell=(2, 2, 2), mesh=(20, 20, 20)):
    """收集力 → 力常数 → 声子谱 → 稳定性判定。"""
    from phonopy import Phonopy, load
    from pymatgen.io.vasp.outputs import Vasprun

    pdir = cid_dir / "phonon"
    ph = load(pdir / "phonopy_disp.yaml")
    disps_root = pdir / "disps"
    disp_dirs = sorted(disps_root.glob("disp_*"))
    force_sets = []
    for d in disp_dirs:
        vr = Vasprun(str(d / "vasprun.xml"), parse_dos=False, parse_eigen=False)
        force_sets.append(np.array(vr.ionic_steps[-1]["forces"]))
    ph.forces = force_sets
    ph.produce_force_constants()

    # 均匀网格上的最小频率 (检查虚频; 虚频在 phonopy 中以负频表示)
    ph.run_mesh(mesh)
    mesh_dict = ph.get_mesh_dict()
    freqs = mesh_dict["frequencies"]          # THz, (nq, nbands)
    min_freq = float(np.min(freqs))
    # 容差: 数值噪声允许约 -0.1 THz (声学支在 Γ 接近 0)
    IMAG_TOL = -0.10
    dynamically_stable = bool(min_freq >= IMAG_TOL)

    # 声子谱 (沿自动路径) 存图
    band_yaml = pdir / "band.yaml"
    try:
        from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
        ph.auto_band_structure(write_yaml=True, filename=str(band_yaml))
    except Exception:
        pass

    result = {
        "cid": cid_dir.name,
        "supercell": list(supercell),
        "n_displacements": len(disp_dirs),
        "min_frequency_THz": min_freq,
        "imaginary_tolerance_THz": IMAG_TOL,
        "dynamically_stable": dynamically_stable,
        "phonon_mesh": list(mesh),
    }
    json.dump(result, open(pdir / "phonon_stability.json", "w"), indent=2)
    print(f"[{cid_dir.name}] 最小声子频率 = {min_freq:.3f} THz | "
          f"动力学稳定 = {dynamically_stable}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Phonon (dynamical) stability of the polar phase")
    ap.add_argument("cid_dir", type=Path)
    ap.add_argument("--supercell", nargs=3, type=int, default=[2, 2, 2])
    ap.add_argument("--disp", type=float, default=0.01)
    ap.add_argument("--collect", action="store_true", help="收集力并判定 (位移计算完成后)")
    args = ap.parse_args()
    if args.collect:
        collect(args.cid_dir, tuple(args.supercell))
    else:
        generate(args.cid_dir, tuple(args.supercell), args.disp)


if __name__ == "__main__":
    main()
