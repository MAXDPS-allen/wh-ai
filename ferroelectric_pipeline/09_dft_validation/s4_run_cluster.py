#!/usr/bin/env python3
"""
Stage 4: 在 GPU 节点上并行运行 VASP (无 SLURM, 直接 SSH)
=====================================================================
把一批 VASP 计算目录分发到在线且空闲的 GPU 节点 (g1-g9) 上运行。
- 探测节点在线状态与空闲显存 (nvidia-smi via ssh)
- 每个计算占用 1 张 GPU (CUDA_VISIBLE_DEVICES), 通过 mpirun 启动 vasp_std
- 简单任务队列: 有空闲 GPU 槽位就派发下一个计算
- 完成判据: OUTCAR 出现 'Total CPU time' / 'Voluntary'; 失败则标记

注意: 本脚本只负责"跑", 输入由 Stage 3 生成, 结果由 Stage 5 解析。
单个 VASP 计算可能耗时数十分钟到数小时, 请在后台运行。

用法:
  python s4_run_cluster.py runs/<cid>/vasp --pattern 'relax_*'    # 先跑弛豫
  python s4_run_cluster.py runs/<cid>/vasp --pattern 'static_*' 'polar_*'
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import config


def ssh(node: str, cmd: str, timeout=30) -> str:
    full = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", node, cmd]
    try:
        return subprocess.run(full, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def probe_gpus():
    """返回 [(node, gpu_index, free_mb), ...] 满足空闲显存阈值的 GPU 槽位。"""
    slots = []
    for node in config.GPU_NODES:
        out = ssh(node, "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits")
        if not out:
            continue
        for line in out.splitlines():
            try:
                idx, free = [x.strip() for x in line.split(",")]
                if int(free) >= config.MIN_FREE_VRAM_MB:
                    slots.append((node, int(idx), int(free)))
            except Exception:
                continue
    return slots


def is_done(calc_dir: Path) -> bool:
    outcar = calc_dir / "OUTCAR"
    if not outcar.exists():
        return False
    try:
        tail = outcar.read_text(errors="ignore")[-2000:]
        return "Total CPU time used" in tail or "Voluntary context switches" in tail
    except Exception:
        return False


def launch(calc_dir: Path, node: str, gpu_idx: int, nproc: int = 1):
    """在 node 的指定 GPU 上后台启动 VASP。返回 ssh Popen。"""
    launcher = config.MPI_LAUNCHER.format(nproc=nproc)
    remote_cmd = (
        f"cd {calc_dir.resolve()} && "
        f"{config.VASP_ENV_SETUP} "
        f"export CUDA_VISIBLE_DEVICES={gpu_idx} && "
        f"{launcher} {config.VASP_STD} > vasp.out 2>&1"
    )
    return subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", node, remote_cmd]
    )


def run(vasp_root: Path, patterns, nproc=1, poll=30, dry_run=False):
    calc_dirs = []
    for pat in patterns:
        calc_dirs += sorted(d for d in vasp_root.glob(pat) if d.is_dir())
    todo = [d for d in calc_dirs if not is_done(d)]
    print(f"待运行: {len(todo)}/{len(calc_dirs)} 个计算 (其余已完成)")

    if dry_run:
        slots = probe_gpus()
        print(f"在线 GPU 槽位: {len(slots)}")
        for n, i, f in slots:
            print(f"  {n} gpu{i}: {f} MB free")
        for d in todo:
            print(f"  would run: {d.name}")
        return

    running = {}    # calc_dir -> (Popen, node, gpu)
    queue = list(todo)
    while queue or running:
        # 回收已结束
        for d in list(running):
            proc, node, gpu = running[d]
            if proc.poll() is not None:
                ok = is_done(d)
                print(f"[{'OK' if ok else 'FAIL'}] {d.name} ({node} gpu{gpu})")
                del running[d]
        # 派发
        if queue:
            slots = probe_gpus()
            busy = {(n, g) for _, n, g in running.values()}
            free_slots = [(n, g) for n, g, _ in slots if (n, g) not in busy]
            while queue and free_slots:
                d = queue.pop(0)
                node, gpu = free_slots.pop(0)
                print(f"-> launch {d.name} on {node} gpu{gpu}")
                running[d] = (launch(d, node, gpu, nproc), node, gpu)
        time.sleep(poll)
    print("全部计算结束。")


def main():
    ap = argparse.ArgumentParser(description="Run VASP calcs across GPU nodes via SSH")
    ap.add_argument("vasp_root", type=Path, help="Stage 3 的 vasp/ 目录")
    ap.add_argument("--pattern", nargs="+", default=["relax_*", "static_*", "polar_*"])
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="只探测节点并列出将运行的计算")
    args = ap.parse_args()
    run(args.vasp_root, args.pattern, args.nproc, args.poll, args.dry_run)


if __name__ == "__main__":
    main()
