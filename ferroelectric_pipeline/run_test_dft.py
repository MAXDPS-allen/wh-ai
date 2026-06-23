#!/usr/bin/env python3
"""
端到端测试驱动: 对准备好的候选跑完整 DFT 铁电验证
=====================================================================
对 test_run/<cid> 下已匹配端点 (polar.json + nonpolar.json) 的候选, 依次:
  relax 两端 → 用弛豫结果重插值 → static (带隙/能量) → 金属性早停 →
  polarization(LCALCPOL) → S5 Berry 相分支跟踪 → Ps/能垒/带隙 + 判定
全程在 GPU 节点并行 (s4_run_cluster)。设计为后台长跑, 断点续跑。
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "09_dft_validation"))
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun

import config, s2_interpolate as s2, s3_make_vasp_inputs as s3, s4_run_cluster as s4
import s5_postprocess as s5


def relaxed(calc_dir: Path) -> Structure:
    c = calc_dir / "CONTCAR"
    if c.exists() and c.stat().st_size > 0:
        return Structure.from_file(str(c))
    return Structure.from_file(str(calc_dir / "POSCAR"))


def band_gap(static_dir: Path) -> float:
    vr = Vasprun(str(static_dir / "vasprun.xml"), parse_dos=False, parse_eigen=True)
    return vr.eigenvalue_band_properties[0]


def run_candidate(cdir: Path, nproc=1):
    cid = cdir.name
    print(f"\n{'='*60}\n[{cid}] 开始 DFT 验证\n{'='*60}", flush=True)
    vasp = cdir / "vasp"

    # 1) 弛豫两端
    s3.run(cdir, stages=("relax",))
    s4.run(vasp, ["relax_*"], nproc=nproc)
    # 2) 用弛豫结果更新端点
    json.dump(relaxed(vasp / "relax_nonpolar").as_dict(), open(cdir / "nonpolar.json", "w"))
    json.dump(relaxed(vasp / "relax_polar").as_dict(), open(cdir / "polar.json", "w"))
    # 3) 重插值
    s2.run(cdir, n_interp=config.N_INTERP)
    # 4) static
    s3.run(cdir, stages=("static",))
    s4.run(vasp, ["static_*"], nproc=nproc)
    # 5) 金属性早停
    gaps = []
    for sd in sorted(vasp.glob("static_image_*")):
        try: gaps.append(band_gap(sd))
        except Exception: gaps.append(0.0)
    print(f"[{cid}] 路径带隙: min={min(gaps):.3f} eV", flush=True)
    if min(gaps) < config.METAL_GAP_EV:
        res = {"cid": cid, "verdict": "metallic_path", "gap_min": min(gaps),
               "is_ferroelectric": False}
        json.dump(res, open(cdir / "dft_validation_result.json", "w"), indent=2)
        print(f"[{cid}] 路径含金属 → 非铁电", flush=True)
        return res
    # 6) polarization + 7) 后处理
    s3.run(cdir, stages=("polar",))
    s4.run(vasp, ["polar_*"], nproc=nproc)
    return s5.run_from_dirs(cdir)


def main():
    out = HERE / "test_run"
    cands = json.load(open(out / "candidates.json"))
    results = []
    for c in cands:
        cdir = out / c["cid"]
        try:
            results.append(run_candidate(cdir))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"cid": c["cid"], "error": str(e)})
    json.dump(results, open(out / "dft_results_summary.json", "w"), indent=2, default=str)
    print(f"\n{'='*60}\n全部完成 -> {out}/dft_results_summary.json\n{'='*60}")
    for r in results:
        print(json.dumps({k: r.get(k) for k in
              ("cid", "Ps_norm_uC_cm2", "gap_polar_eV", "path_barrier_meV",
               "is_ferroelectric", "is_high_quality", "verdict")}, default=str))


if __name__ == "__main__":
    main()
