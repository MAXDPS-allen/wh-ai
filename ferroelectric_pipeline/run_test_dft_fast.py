#!/usr/bin/env python3
"""
端到端测试驱动 (冻结畸变版, 跳过弛豫)
=====================================================================
MP 极性结构已是 DFT 弛豫态; 非极性参考由 in-cell 对称化得到 (同晶胞)。
因此采用"冻结畸变"图像: 固定晶胞, 沿 非极性→极性 畸变路径计算能量/带隙/
Berry 相极化 —— 这正是铁电软模/双势阱的标准刻画, 且避免 ISIF=3 小胞振荡。

流程: 插值(同胞) → static(能量+带隙) → 金属性早停 → polarization(LCALCPOL)
       → S5 分支跟踪 → Ps/能垒/带隙 + 判定。GPU 并行, 后台长跑。
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "09_dft_validation"))
from pymatgen.io.vasp.outputs import Vasprun
import config, s2_interpolate as s2, s3_make_vasp_inputs as s3
import s4_run_cluster as s4, s5_postprocess as s5


def band_gap(static_dir: Path) -> float:
    vr = Vasprun(str(static_dir / "vasprun.xml"), parse_dos=False, parse_eigen=True)
    return vr.eigenvalue_band_properties[0]


def run_candidate(cdir: Path, nproc=1):
    cid = cdir.name
    print(f"\n{'='*60}\n[{cid}] DFT 验证 (冻结畸变)\n{'='*60}", flush=True)
    vasp = cdir / "vasp"
    # 1) 插值 (端点已就位, 同晶胞)
    s2.run(cdir, n_interp=config.N_INTERP)
    # 2) static
    s3.run(cdir, stages=("static",))
    s4.run(vasp, ["static_*"], nproc=nproc)
    gaps = []
    for sd in sorted(vasp.glob("static_image_*")):
        try: gaps.append(band_gap(sd))
        except Exception: gaps.append(0.0)
    print(f"[{cid}] 路径带隙 min={min(gaps):.3f} eV", flush=True)
    if min(gaps) < config.METAL_GAP_EV:
        res = {"cid": cid, "verdict": "metallic_path", "gap_min": min(gaps),
               "is_ferroelectric": False}
        json.dump(res, open(cdir / "dft_validation_result.json", "w"), indent=2)
        print(f"[{cid}] 路径含金属 → 非铁电", flush=True)
        return res
    # 3) polarization + 4) 后处理
    s3.run(cdir, stages=("polar",))
    s4.run(vasp, ["polar_*"], nproc=nproc)
    return s5.run_from_dirs(cdir)


def main():
    out = HERE / "test_run"
    cands = json.load(open(out / "candidates.json"))
    results = []
    for c in cands:
        try:
            results.append(run_candidate(out / c["cid"]))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"cid": c["cid"], "error": str(e)})
    json.dump(results, open(out / "dft_results_summary.json", "w"), indent=2, default=str)
    print(f"\n{'='*60}\n完成 -> {out}/dft_results_summary.json")
    for r in results:
        print(json.dumps({k: r.get(k) for k in
              ("cid", "Ps_norm_uC_cm2", "gap_polar_eV", "dw_depth_meV", "path_barrier_meV",
               "is_ferroelectric", "is_high_quality", "verdict")}, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
