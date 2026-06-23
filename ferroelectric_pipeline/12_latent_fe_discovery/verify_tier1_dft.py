#!/usr/bin/env python3
"""
DFT 验证 Tier-1 候选: DFPT Γ 声子 → 确认是否真有软模 (虚频)
=====================================================================
对 Tier-1 打分的候选 (含若干高分"预测失稳" + 若干低分"预测稳定"对照),
在 MP 高对称几何上做 DFPT (IBRION=8) Γ 声子计算, 解析虚频数目。
预测失稳者应有虚频; 对照应无 → 验证 Tier-1 判别器。

阶段: (1) 生成 DFPT 输入 (2) CPU 构建 g 节点运行 (3) 解析 OUTCAR Γ 频率。
用法:
  conda run -n fe_dft python verify_tier1_dft.py --prepare
  python verify_tier1_dft.py --run         # 在 g 节点 CPU 跑
  conda run -n fe_dft python verify_tier1_dft.py --parse
"""
from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "09_dft_validation"))
DFT_DIR = HERE / "tier1_screen" / "dft"

# 验证集: 2 个高分(预测失稳) + 1 个低分(预测稳定)对照
CANDIDATES = [
    ("BCl3_mp-23184", "unstable?", 1.0),
    ("Ba2CaMoO6_mp-19403", "unstable?", 1.0),
    ("Ac2O3_mp-11107", "stable(control)", 0.0),
]


def prepare():
    import s3_make_vasp_inputs as s3
    from pymatgen.core import Structure
    s3._configure_potcar()
    DFT_DIR.mkdir(parents=True, exist_ok=True)
    screen = HERE / "tier1_screen"
    for cid, _, _ in CANDIDATES:
        # 结构: 高分候选在 tier1_screen/<cid>/polar.json; 对照需现取
        src = screen / cid / "polar.json"
        if not src.exists():
            # 从 ranked + 重新取结构 (对照不在 top dirs); 用 MP 重取
            from mp_api.client import MPRester
            mid = cid.split("_")[-1]
            with MPRester("1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq") as m:
                st = m.materials.summary.search(material_ids=[mid], fields=["structure"])[0].structure
        else:
            st = Structure.from_dict(json.load(open(src)))
        dest = DFT_DIR / cid
        s3.write_dfpt_gamma(st, dest)
        print(f"[{cid}] DFPT inputs -> {dest} ({len(st)} atoms)", flush=True)


def run():
    import s4_run_cluster as s4
    s4.run(DFT_DIR, ["*"], nproc=1, poll=20, backend="cpu")


def parse_outcar_freqs(outcar: Path):
    """解析 OUTCAR 中 Γ 声子频率, 返回 (n_imaginary, min_freq_THz)。
    VASP DFPT 打印: '   1 f  =   ... THz ...' (实), '   k f/i=  ... THz' (虚)。"""
    txt = outcar.read_text(errors="ignore")
    freqs = []
    for m in re.finditer(r"\b(\d+)\s+f(/i|\s)\s*=\s*([\-0-9.]+)\s*THz", txt):
        imag = (m.group(2) == "/i")
        val = float(m.group(3))
        freqs.append(-val if imag else val)
    return freqs


def parse():
    res = []
    for cid, expect, score in CANDIDATES:
        oc = DFT_DIR / cid / "OUTCAR"
        if not oc.exists():
            res.append({"cid": cid, "status": "no OUTCAR"}); continue
        fr = parse_outcar_freqs(oc)
        if not fr:
            res.append({"cid": cid, "status": "no freqs parsed"}); continue
        nimag = sum(1 for f in fr if f < -0.1)
        res.append({"cid": cid, "tier1_score": score, "expectation": expect,
                    "n_imaginary": nimag, "min_freq_THz": round(min(fr), 3),
                    "has_soft_mode": nimag > 0,
                    "DFT_confirms_tier1": (nimag > 0) == (score > 0.5)})
    json.dump(res, open(DFT_DIR / "verification.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    ok = sum(1 for r in res if r.get("DFT_confirms_tier1"))
    print(f"\nDFT confirms Tier-1 on {ok}/{len(res)} candidates", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--parse", action="store_true")
    args = ap.parse_args()
    if args.prepare: prepare()
    if args.run: run()
    if args.parse: parse()


if __name__ == "__main__":
    main()
