#!/usr/bin/env python3
"""
从 Petretto DDB 文件本地提取 Γ 点声子本征位移 (无需 API, 绕开 flaky SSL)
=====================================================================
用 abipy + anaddb 对每个 DDB 计算 Γ 点声子模, 提取本征位移 + 频率 + 结构,
存成与 fetch_phonon_modes 相同的格式, 供 pretrain_modes.py 使用。

需要 conda env `abi` (abipy + abinit/anaddb)。
用法: conda run -n abi python extract_ddb_modes.py --target 1521
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

HERE = Path(__file__).parent


def extract_one(ddb_path: Path, out: Path):
    mid = ddb_path.stem.replace("_DDB", "").replace("out_", "")
    f = out / f"{mid}.npz"
    if f.exists():
        return "skip"
    try:
        from abipy.dfpt.ddb import DdbFile
        ddb = DdbFile(str(ddb_path))
        # 在 Γ 计算声子模 (anaddb), 含本征位移
        phb = ddb.anaget_phmodes_at_qpoint(qpoint=(0, 0, 0), lo_to_splitting=False)
        freqs = np.array(phb.phfreqs)[0] * 33.35641        # eV->THz? phfreqs in eV; ->THz
        # abipy phfreqs 单位 eV; 转 THz: 1 eV = 241.799 THz. 用 THz 便于一致
        freqs = np.array(phb.phfreqs)[0] * 241.79893        # (nmodes,) THz
        disp = np.array(phb.phdispl_cart)[0]                # (nmodes, natom*3) complex
        natom = disp.shape[1] // 3
        edg = np.real(disp).reshape(len(freqs), natom, 3).astype(np.float32)
        struct = phb.structure.as_dict()
        np.savez_compressed(f, structure=json.dumps(struct),
                            gamma_freqs=freqs.astype(np.float32), gamma_eigdisp=edg)
        ddb.close()
        return "ok"
    except Exception as e:
        return "err:" + str(e)[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddb_dir", type=Path, default=HERE / "petretto" / "ddbs")
    ap.add_argument("--out", type=Path, default=HERE / "modes_data")
    ap.add_argument("--target", type=int, default=1521)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ddbs = sorted(args.ddb_dir.rglob("*DDB*")) or sorted(args.ddb_dir.rglob("*.ddb")) \
        or sorted(p for p in args.ddb_dir.rglob("*") if p.is_file() and "DDB" in p.name)
    ddbs = ddbs[:args.target]
    print(f"found {len(ddbs)} DDB files", flush=True)
    ok = err = skip = 0
    for i, d in enumerate(ddbs):
        s = extract_one(d, args.out)
        if s == "ok": ok += 1
        elif s == "skip": skip += 1
        else:
            err += 1
            if err <= 3: print("  sample err:", s, flush=True)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ddbs)} | ok={ok} skip={skip} err={err}", flush=True)
    print(f"DONE: ok={ok} skip={skip} err={err} | total modes saved={len(list(args.out.glob('*.npz')))}",
          flush=True)


if __name__ == "__main__":
    main()
