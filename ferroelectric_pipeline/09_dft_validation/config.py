#!/usr/bin/env python3
"""
DFT 验证管线配置 (集群相关路径与参数)
=====================================================================
集中管理 VASP 二进制、POTCAR 库、GPU 节点、以及复刻 Smidt et al. 2020
论文的 DFT 收敛参数。修改本文件以适配不同集群。
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 集群环境 (本集群: 无 SLURM/PBS, 直接 SSH 到 GPU 节点)
# ---------------------------------------------------------------------------
# VASP 6.4.2 GPU (CUDA) 构建
VASP_STD = "/share/apps/vasp.6.4.2_nvhpc2311_hpcx_cuda11.8/bin/vasp_std"
# VASP PAW PBE 赝势库。pymatgen 期望布局: <PSP_DIR>/POT_GGA_PAW_PBE/<symbol>/POTCAR
# 集群原始库在 /share/apps/vasp/potentials/potpaw_PBE, 通过 psp_mirror/ 软链接成
# pymatgen 期望的命名 (POT_GGA_PAW_PBE -> potpaw_PBE)。见 environment/setup_psp.sh
VASP_PSP_DIR = str(Path(__file__).parent / "psp_mirror")
POTCAR_FUNCTIONAL = "PBE"

# 可用 GPU 计算节点 (免密 SSH)。运行前会探测在线与空闲显存。
GPU_NODES = ["g3", "g4", "g6", "g1", "g7", "g2", "g5", "g8", "g9"]
GPU_PER_NODE = 2                 # 每节点可用 GPU 数 (按需调整)
MIN_FREE_VRAM_MB = 8000          # 派发任务所需的最小空闲显存

# 启动 VASP 前需 source 的环境 (nvhpc + cuda + hpcx)
VASP_ENV_SETUP = (
    "module load 2>/dev/null; "
    "export OMP_NUM_THREADS=4; "
)
# MPI 启动器 (GPU 版按 GPU 数选择 rank)
MPI_LAUNCHER = "mpirun -np {nproc}"

# conda 环境 (pymatgen 后处理)
CONDA_SETUP = "source /share/home/caiby/miniforge3/etc/profile.d/conda.sh && conda activate fe_dft"

# 工作根目录
WORK_ROOT = Path(__file__).parent / "runs"

# ---------------------------------------------------------------------------
# DFT 参数 (复刻 Smidt et al. 2020, Sci. Data 7:72)
# ---------------------------------------------------------------------------
ENCUT = 520                      # eV, 1.3× 最大推荐截断
EDIFF_PER_ATOM = 5e-5            # eV/atom, 电子收敛
EDIFFG_PER_ATOM = -5e-4          # eV/Å, 离子收敛 (力判据, 取负号)
KPT_DENSITY_RELAX = 50           # k 点密度 /(1/Å)³, 弛豫
KPT_DENSITY_STATIC = 100         # k 点密度 /(1/Å)³, 静态/极化
ISPIN = 2                        # 自旋极化, 初始铁磁
N_INTERP = 8                     # 非极性↔极性间线性插值数
METAL_GAP_EV = 0.01              # 带隙 < 10 meV 判为金属 → 终止
SYMPREC = 0.1                    # 对称性精度 (Å), 与 Materials Project 一致

# 质量判据 (论文)
POL_SMOOTH_TOL = 0.1             # 极化-样条偏差 μC/cm²
ENERGY_SMOOTH_TOL = 0.01         # 能量-样条偏差 eV


def check_paths(verbose=True):
    """校验关键路径是否存在。"""
    ok = True
    for label, p in [("vasp_std", VASP_STD),
                     ("POTCAR PBE", str(Path(VASP_PSP_DIR) / "POT_GGA_PAW_PBE"))]:
        exists = Path(p).exists()
        ok &= exists
        if verbose:
            print(f"  [{'OK' if exists else 'MISSING'}] {label}: {p}")
    return ok


if __name__ == "__main__":
    print("DFT validation pipeline config:")
    check_paths()
