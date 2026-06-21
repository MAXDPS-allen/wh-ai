#!/usr/bin/env python3
"""
Stage 3: 生成各阶段 VASP 输入 (复刻 Smidt et al. 2020 参数)
=====================================================================
计算类型:
  relax    两端 (非极性/极性) 全弛豫 (ISIF=3, 原子+晶胞)
  static   路径上 10 个结构的自洽 + 带隙
  polar    路径上 10 个结构的 Berry 相极化 (LCALCPOL=.TRUE.)

参数 (论文 + Materials Project 规范):
  PBE-GGA + PAW; ENCUT=520; EDIFF=5e-5/atom; EDIFFG=-5e-4/atom;
  ISPIN=2 初始铁磁; PBE+U (MP 规则, 由 MPRelaxSet 自动套用);
  k 点密度: 弛豫 50, 静态/极化 100 /(1/Å)³。

依赖: pymatgen (MPRelaxSet/MPStaticSet 提供 MP 一致的 U/POTCAR/MAGMOM)
注意: 需配置 POTCAR 路径 (config.VASP_PSP_DIR), 由本脚本自动写入 pymatgen 设置。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet

import config


def _configure_potcar():
    """让 pymatgen 找到 POTCAR 库。"""
    os.environ.setdefault("PMG_VASP_PSP_DIR", config.VASP_PSP_DIR)
    try:
        from pymatgen.core import SETTINGS
        SETTINGS["PMG_VASP_PSP_DIR"] = config.VASP_PSP_DIR
        SETTINGS["PMG_DEFAULT_FUNCTIONAL"] = "PBE"
    except Exception:
        pass


def _kpoints_for(struct: Structure, density: int):
    from pymatgen.io.vasp.inputs import Kpoints
    return Kpoints.automatic_density(struct, density)


def write_relax(struct: Structure, dest: Path):
    incar = {
        "ENCUT": config.ENCUT,
        "EDIFF_PER_ATOM": config.EDIFF_PER_ATOM,
        "EDIFFG": config.EDIFFG_PER_ATOM,
        "ISPIN": config.ISPIN,
        "IBRION": 2, "ISIF": 3, "NSW": 99,
        "LCHARG": False, "LWAVE": False,
        "LREAL": "Auto",
    }
    vis = MPRelaxSet(struct, user_incar_settings=incar,
                     user_kpoints_settings={"reciprocal_density": config.KPT_DENSITY_RELAX})
    vis.write_input(str(dest))


def write_static(struct: Structure, dest: Path):
    incar = {
        "ENCUT": config.ENCUT,
        "EDIFF_PER_ATOM": config.EDIFF_PER_ATOM,
        "ISPIN": config.ISPIN,
        "NSW": 0, "IBRION": -1,
        "LCHARG": False, "LWAVE": False,
        "LORBIT": 11,
    }
    vis = MPStaticSet(struct, user_incar_settings=incar,
                      user_kpoints_settings={"reciprocal_density": config.KPT_DENSITY_STATIC})
    vis.write_input(str(dest))


def write_polarization(struct: Structure, dest: Path):
    """Berry 相极化: LCALCPOL=.TRUE. 自洽计算。"""
    incar = {
        "ENCUT": config.ENCUT,
        "EDIFF_PER_ATOM": config.EDIFF_PER_ATOM,
        "ISPIN": config.ISPIN,
        "NSW": 0, "IBRION": -1,
        "LCALCPOL": True,         # 触发 Berry 相极化计算
        "LCHARG": False, "LWAVE": False,
    }
    vis = MPStaticSet(struct, user_incar_settings=incar,
                      user_kpoints_settings={"reciprocal_density": config.KPT_DENSITY_STATIC})
    vis.write_input(str(dest))


def run(cid_dir: Path, stages=("relax", "static", "polar")):
    _configure_potcar()
    path_dir = cid_dir / "path"
    vasp_root = cid_dir / "vasp"
    vasp_root.mkdir(exist_ok=True)
    made = []

    if "relax" in stages:
        for end, name in [("nonpolar", "image_00"), ("polar", None)]:
            st = Structure.from_dict(json.load(open(cid_dir / f"{end}.json")))
            dest = vasp_root / f"relax_{end}"
            write_relax(st, dest)
            made.append(str(dest))

    images = sorted(path_dir.glob("image_*.json"))
    if "static" in stages:
        for img in images:
            st = Structure.from_dict(json.load(open(img)))
            dest = vasp_root / f"static_{img.stem}"
            write_static(st, dest)
            made.append(str(dest))
    if "polar" in stages:
        for img in images:
            st = Structure.from_dict(json.load(open(img)))
            dest = vasp_root / f"polar_{img.stem}"
            write_polarization(st, dest)
            made.append(str(dest))

    json.dump({"calcs": made}, open(vasp_root / "inputs_manifest.json", "w"), indent=2)
    print(f"[{cid_dir.name}] 生成 {len(made)} 个 VASP 计算目录 -> {vasp_root}")
    return made


def main():
    ap = argparse.ArgumentParser(description="Write VASP inputs for relax/static/polarization")
    ap.add_argument("cid_dir", type=Path)
    ap.add_argument("--stages", nargs="+", default=["relax", "static", "polar"])
    args = ap.parse_args()
    run(args.cid_dir, tuple(args.stages))


if __name__ == "__main__":
    main()
