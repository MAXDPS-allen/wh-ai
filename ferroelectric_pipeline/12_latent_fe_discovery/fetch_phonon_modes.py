#!/usr/bin/env python3
"""
拉取 MP 声子库 Γ 点本征位移 (软模发现头的训练信号)
=====================================================================
MP 声子 summary 不含本征矢, 但 get_phonon_bandstructure_by_material_id 返回
eigendisplacements (nbands, nq, natoms, 3)。本脚本并行拉取小胞材料的 Γ 点
本征位移 + 频率 + 结构, 增量保存 (可断点续传)。

每材料保存 modes_data/<id>.npz: structure(json), gamma_freqs, gamma_eigdisp(real)
慢 (~30s/材料), 用线程并发 + 增量保存。
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

KEY = "1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq"
_M = None
_OUT = None


def _init(out_str):
    """工作进程只设输出目录; MPRester 在每次 fetch_one 内新建 (fork 后才建, 连接新鲜)。"""
    global _OUT
    _OUT = Path(out_str)


class _Timeout(Exception):
    pass


def fetch_one(mid, timeout=240):
    import signal
    out = _OUT
    f = out / f"{mid}.npz"
    if f.exists():
        return ("skip", mid)

    def _handler(signum, frame):
        raise _Timeout()
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        from mp_api.client import MPRester
        bs = None
        for attempt in range(15):                 # 每次用全新 MPRester (避免长连接 SSL 失效)
            try:
                with MPRester(KEY) as mr:
                    bs = mr.get_phonon_bandstructure_by_material_id(mid)
                break
            except _Timeout:
                raise
            except Exception:
                if attempt == 14:
                    raise
                time.sleep(min(1.0 + attempt * 0.5, 6.0))
        q = np.array(bs.qpoints)
        gi = int(np.argmin(np.linalg.norm(q, axis=1)))      # Gamma index
        fr = np.array(bs.frequencies)[:, gi]                 # (nbands,) THz
        ed = np.array(bs.eigendisplacements)                 # (nbands, nq, natoms, 3) complex
        edg = np.real(ed[:, gi, :, :]).astype(np.float32)    # (nbands, natoms, 3) Gamma, real part
        struct = bs.structure.as_dict() if hasattr(bs, "structure") and bs.structure else \
                 bs.to_pmg().structure.as_dict()
        np.savez_compressed(f, structure=json.dumps(struct),
                            gamma_freqs=fr.astype(np.float32), gamma_eigdisp=edg)
        return ("ok", mid)
    except _Timeout:
        return ("err:timeout", mid)
    except Exception as e:
        return ("err:" + str(e)[:40], mid)
    finally:
        signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/share/home/caiby/mp_datasets/build/static-collections/phonon")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "modes_data")
    ap.add_argument("--max_nsites", type=int, default=20)
    ap.add_argument("--target", type=int, default=800)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from deltalake import DeltaTable
    dt = DeltaTable(args.cache)
    tbl = dt.to_pyarrow_table(columns=["identifier", "nsites", "structure"])
    ids = tbl.column("identifier").to_pylist()
    nst = tbl.column("nsites").to_pylist()
    # 选小胞材料
    cand = [ids[i] for i in range(len(ids)) if nst[i] is not None and 0 < nst[i] <= args.max_nsites]
    cand = cand[:args.target]
    done = {p.stem for p in args.out.glob("*.npz")}
    todo = [c for c in cand if c not in done]
    print(f"target {len(cand)} small-cell materials | already have {len(done)} | to fetch {len(todo)}", flush=True)

    # 顺序拉取 (每次新建 MPRester)。并发会触发 MP 端 SSL/连接限制 -> 全失败, 故串行。
    _init(str(args.out))
    ok = err = 0
    t0 = time.time()
    for i, mid in enumerate(todo):
        status, _ = fetch_one(mid)
        if status == "ok": ok += 1
        elif status.startswith("err"): err += 1
        if (i + 1) % 10 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(todo)} | ok={ok} err={err} | {rate*60:.1f}/min | "
                  f"total saved={len(done)+ok} | ~{(len(todo)-i-1)/max(rate,1e-9)/60:.0f}min left",
                  flush=True)
    print(f"DONE: fetched {ok}, errors {err}, total saved {len(done)+ok}", flush=True)


if __name__ == "__main__":
    main()
