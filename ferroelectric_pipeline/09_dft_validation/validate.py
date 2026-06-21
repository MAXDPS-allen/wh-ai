#!/usr/bin/env python3
"""
DFT 验证管线编排器 (端到端)
=====================================================================
把一个 ML 候选 (极性结构, 可选非极性母相) 跑完整的第一性原理铁电验证,
复刻 Smidt et al. 2020 的工作流顺序:

  S1  找/校验非极性母相 (群-子群)                         [s1]
  S3a 生成两端弛豫输入                                     [s3 relax]
  S4a 在 GPU 节点上跑弛豫                                  [s4]
  ──  用弛豫后的两端更新 polar/nonpolar
  S2  在弛豫后的两端之间生成 8 个插值                       [s2]
  ──  金属性早停: 任一端点/插值带隙 < 10 meV 则终止
  S3b 生成路径的 static + polarization(LCALCPOL) 输入       [s3 static/polar]
  S4b 在 GPU 节点上跑 static + polarization                [s4]
  S5  Berry 相分支跟踪 → Ps/能垒/带隙 + 质量判定           [s5]

阶段较长 (VASP 计算耗时), 支持断点续跑: 已完成的计算 (OUTCAR 收敛) 自动跳过。
单步也可手动调用各 sN 脚本。

用法:
  conda activate fe_dft
  python validate.py candidate_polar.json --nonpolar candidate_nonpolar.json --cid mycand
  # 仅准备输入, 不自动跑 VASP (手动控制集群):
  python validate.py candidate_polar.json --prepare-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun, Outcar

import config
import s1_find_nonpolar_parent as s1
import s2_interpolate as s2
import s3_make_vasp_inputs as s3
import s4_run_cluster as s4


def _relaxed_structure(calc_dir: Path) -> Structure:
    """从弛豫目录读取弛豫后的结构 (优先 CONTCAR)。"""
    contcar = calc_dir / "CONTCAR"
    if contcar.exists() and contcar.stat().st_size > 0:
        return Structure.from_file(str(contcar))
    return Structure.from_file(str(calc_dir / "POSCAR"))


def _band_gap(static_dir: Path) -> float:
    vr = Vasprun(str(static_dir / "vasprun.xml"), parse_dos=False, parse_eigen=True)
    return vr.eigenvalue_band_properties[0]


def validate(polar_path: Path, out_dir: Path, nonpolar_path=None, cid=None,
             prepare_only=False, nproc=1):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- S1: 非极性母相 ----
    info = s1.run(polar_path, out_dir, nonpolar_path, cid)
    cid = cid or info.get("cid") or Structure.from_dict(
        json.load(open(polar_path))).composition.reduced_formula
    cdir = out_dir / cid
    if not info.get("valid", False):
        print(f"[{cid}] 不满足铁电必要条件 (无合法非极性母相), 终止。")
        return {"cid": cid, "stage": "S1", "verdict": "not_ferroelectric_candidate"}

    # ---- S3a + S4a: 弛豫两端 ----
    s3.run(cdir, stages=("relax",))
    if prepare_only:
        print(f"[{cid}] 已生成弛豫输入。手动运行: "
              f"python s4_run_cluster.py {cdir/'vasp'} --pattern 'relax_*'")
        return {"cid": cid, "stage": "S3a", "verdict": "inputs_ready"}
    s4.run(cdir / "vasp", ["relax_*"], nproc=nproc)

    # ---- 用弛豫结果更新两端 ----
    nonpolar_relaxed = _relaxed_structure(cdir / "vasp" / "relax_nonpolar")
    polar_relaxed = _relaxed_structure(cdir / "vasp" / "relax_polar")
    json.dump(nonpolar_relaxed.as_dict(), open(cdir / "nonpolar.json", "w"))
    json.dump(polar_relaxed.as_dict(), open(cdir / "polar.json", "w"))

    # ---- S2: 插值 ----
    s2.run(cdir, n_interp=config.N_INTERP)

    # ---- S3b + S4b: static + polarization ----
    s3.run(cdir, stages=("static", "polar"))
    s4.run(cdir / "vasp", ["static_*"], nproc=nproc)

    # ---- 金属性早停 ----
    gaps = []
    for sd in sorted((cdir / "vasp").glob("static_image_*")):
        try:
            gaps.append(_band_gap(sd))
        except Exception:
            gaps.append(0.0)
    if min(gaps) < config.METAL_GAP_EV:
        print(f"[{cid}] 路径含金属结构 (min gap={min(gaps):.3f} eV < "
              f"{config.METAL_GAP_EV}), 非铁电, 终止。")
        res = {"cid": cid, "stage": "S4b", "verdict": "metallic", "gap_min": min(gaps)}
        json.dump(res, open(cdir / "dft_validation_result.json", "w"), indent=2)
        return res

    # ---- S4b: polarization + S5 后处理 ----
    s4.run(cdir / "vasp", ["polar_*"], nproc=nproc)
    import s5_postprocess as s5
    return s5.run_from_dirs(cdir)


def main():
    ap = argparse.ArgumentParser(description="End-to-end DFT ferroelectric validation")
    ap.add_argument("polar", type=Path)
    ap.add_argument("--nonpolar", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=config.WORK_ROOT)
    ap.add_argument("--cid", default=None)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--nproc", type=int, default=1)
    args = ap.parse_args()
    validate(args.polar, args.out, args.nonpolar, args.cid, args.prepare_only, args.nproc)


if __name__ == "__main__":
    main()
