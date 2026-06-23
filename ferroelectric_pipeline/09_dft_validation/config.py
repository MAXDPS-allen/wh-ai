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
GPU_NODES = ["g3", "g4", "g6", "g1", "g7"]  # g8 lacks AVX512 (VASP SIGILL); g2/g5/g9 offline
GPU_PER_NODE = 2                 # 每节点可用 GPU 数 (按需调整)
MIN_FREE_VRAM_MB = 8000          # 派发任务所需的最小空闲显存

# 启动 VASP 前的环境 (nvhpc 2311 + cuda 11.8 + hpcx ompi)。
# 完全显式设置所有库路径, 不依赖 hpcx-init.sh/hpcx_load (其跨节点行为不一致,
# 曾导致部分计算因找不到 MPI/qd 库而以 exit 127 失败)。
_NV = "/share/apps/nvhpc/2311/Linux_x86_64/23.11"
_HPCX = f"{_NV}/comm_libs/11.8/hpcx/hpcx-2.14"
_LIBS = ":".join([
    f"{_NV}/compilers/lib",
    f"{_NV}/compilers/extras/qd/lib",
    f"{_NV}/cuda/11.8/lib64",
    f"{_NV}/math_libs/11.8/lib64",
    f"{_HPCX}/ompi/lib",
    f"{_HPCX}/ucx/lib",
    f"{_HPCX}/ucc/lib",
])
# 部分计算节点缺少系统库 libatomic.so.1; conda 环境的 lib 在共享盘上, 追加到
# LD_LIBRARY_PATH 末尾补齐 (放末尾避免覆盖系统/nvhpc 的 libstdc++ 等)。
_CONDA_LIB = "/share/home/caiby/miniforge3/envs/fe_dft/lib"
VASP_ENV_SETUP = (
    f"export PATH={_HPCX}/ompi/bin:{_NV}/compilers/bin:$PATH; "
    f"export LD_LIBRARY_PATH={_LIBS}:$LD_LIBRARY_PATH:{_CONDA_LIB}; "
    "export OMP_NUM_THREADS=4; "
    "export OMPI_MCA_btl_openib_allow_ib=0 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader; "
)
# MPI 启动器 (hpcx mpirun 全路径; GPU 版每 rank 绑一张卡)
_MPIRUN = f"{_HPCX}/ompi/bin/mpirun"
MPI_LAUNCHER = _MPIRUN + " --allow-run-as-root -np {nproc}"

# ---------------------------------------------------------------------------
# CPU 后端 (用于 Berry 相极化 LCALCPOL)
# ---------------------------------------------------------------------------
# GPU (OpenACC) 构建的 VASP 在 LCALCPOL/Berry 相处挂起 (该特性 GPU 端不支持),
# 故极化阶段改用 CPU 构建 + hpcx mpirun -np 1 (单 rank, 小胞足够; 多 rank 需
# 完整 UCX PML 配置, 跨节点不稳)。仅在 AVX512 节点运行 (无 AVX512 节点会 SIGILL)。
VASP_STD_CPU = "/share/apps/nvhpc_vasp/vasp.6.4.1/bin/vasp_std"
CPU_NODES = ["g3", "g4", "g1", "g6", "g7"]     # 均含 avx512f
VASP_ENV_SETUP_CPU = (
    f"source {_HPCX}/hpcx-init.sh && hpcx_load 2>/dev/null; "
    f"export LD_LIBRARY_PATH={_NV}/compilers/lib:{_NV}/compilers/extras/qd/lib:"
    f"$LD_LIBRARY_PATH:{_CONDA_LIB}; "
    "export OMP_NUM_THREADS=1; "
)
MPI_LAUNCHER_CPU = "mpirun -np {nproc}"         # 在 hpcx_load 后 mpirun 已在 PATH

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
