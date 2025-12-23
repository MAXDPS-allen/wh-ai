"""
统一特征工程模块 - 共享特征提取器
=============================================
所有模型 (GCNN, 逆向设计, GAN) 共享此特征工程模块
确保特征维度和计算方式的一致性

特征维度: 64维
- 晶格参数特征: 12维
- 化学成分特征: 24维  
- 对称性特征: 8维
- 配位/键合特征: 12维
- 热力学特征: 8维
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# ==========================================
# 1. 完整元素数据库
# ==========================================
# [原子序数, 质量, 半径, 电负性, 电离能, 价电子, 熔点(K), 热导率]
ELEMENT_DATABASE = {
    'H':  [1, 1.008, 0.37, 2.20, 13.598, 1, 14.01, 0.18],
    'He': [2, 4.003, 0.32, 0.00, 24.587, 0, 0.95, 0.15],
    'Li': [3, 6.94, 1.34, 0.98, 5.392, 1, 453.65, 84.8],
    'Be': [4, 9.012, 0.90, 1.57, 9.323, 2, 1560, 201],
    'B':  [5, 10.81, 0.82, 2.04, 8.298, 3, 2349, 27.4],
    'C':  [6, 12.011, 0.77, 2.55, 11.260, 4, 3823, 129],
    'N':  [7, 14.007, 0.75, 3.04, 14.534, 5, 63.15, 0.026],
    'O':  [8, 15.999, 0.73, 3.44, 13.618, 6, 54.36, 0.027],
    'F':  [9, 18.998, 0.71, 3.98, 17.423, 7, 53.53, 0.028],
    'Ne': [10, 20.18, 0.69, 0.00, 21.565, 0, 24.56, 0.049],
    'Na': [11, 22.990, 1.54, 0.93, 5.139, 1, 370.95, 142],
    'Mg': [12, 24.305, 1.30, 1.31, 7.646, 2, 923, 156],
    'Al': [13, 26.982, 1.18, 1.61, 5.986, 3, 933.47, 237],
    'Si': [14, 28.085, 1.11, 1.90, 8.152, 4, 1687, 149],
    'P':  [15, 30.974, 1.06, 2.19, 10.487, 5, 317.3, 0.24],
    'S':  [16, 32.06, 1.02, 2.58, 10.360, 6, 388.36, 0.21],
    'Cl': [17, 35.45, 0.99, 3.16, 12.968, 7, 171.6, 0.009],
    'Ar': [18, 39.948, 0.97, 0.00, 15.760, 0, 83.81, 0.018],
    'K':  [19, 39.098, 1.96, 0.82, 4.341, 1, 336.53, 102.5],
    'Ca': [20, 40.078, 1.74, 1.00, 6.113, 2, 1115, 201],
    'Sc': [21, 44.956, 1.44, 1.36, 6.561, 3, 1814, 15.8],
    'Ti': [22, 47.867, 1.36, 1.54, 6.828, 4, 1941, 21.9],
    'V':  [23, 50.942, 1.25, 1.63, 6.746, 5, 2183, 30.7],
    'Cr': [24, 51.996, 1.27, 1.66, 6.767, 6, 2180, 93.9],
    'Mn': [25, 54.938, 1.39, 1.55, 7.434, 7, 1519, 7.81],
    'Fe': [26, 55.845, 1.25, 1.83, 7.902, 8, 1811, 80.4],
    'Co': [27, 58.933, 1.26, 1.88, 7.881, 9, 1768, 100],
    'Ni': [28, 58.693, 1.21, 1.91, 7.640, 10, 1728, 90.9],
    'Cu': [29, 63.546, 1.38, 1.90, 7.726, 11, 1357.77, 401],
    'Zn': [30, 65.38, 1.31, 1.65, 9.394, 12, 692.68, 116],
    'Ga': [31, 69.723, 1.26, 1.81, 6.000, 3, 302.91, 40.6],
    'Ge': [32, 72.63, 1.22, 2.01, 7.900, 4, 1211.40, 60.2],
    'As': [33, 74.922, 1.19, 2.18, 9.789, 5, 1090, 50.2],
    'Se': [34, 78.96, 1.16, 2.55, 9.752, 6, 494, 0.52],
    'Br': [35, 79.904, 1.14, 2.96, 11.814, 7, 265.8, 0.12],
    'Kr': [36, 83.798, 1.10, 3.00, 14.000, 0, 115.79, 0.009],
    'Rb': [37, 85.468, 2.11, 0.82, 4.177, 1, 312.46, 58.2],
    'Sr': [38, 87.62, 1.92, 0.95, 5.695, 2, 1050, 35.4],
    'Y':  [39, 88.906, 1.62, 1.22, 6.217, 3, 1799, 17.2],
    'Zr': [40, 91.224, 1.48, 1.33, 6.634, 4, 2128, 22.6],
    'Nb': [41, 92.906, 1.37, 1.60, 6.759, 5, 2750, 53.7],
    'Mo': [42, 95.96, 1.45, 2.16, 7.092, 6, 2896, 138],
    'Tc': [43, 98.0, 1.56, 1.90, 7.28, 7, 2430, 50.6],
    'Ru': [44, 101.07, 1.26, 2.20, 7.361, 8, 2607, 117],
    'Rh': [45, 102.91, 1.35, 2.28, 7.459, 9, 2237, 150],
    'Pd': [46, 106.42, 1.31, 2.20, 8.337, 10, 1828.05, 71.8],
    'Ag': [47, 107.87, 1.53, 1.93, 7.576, 11, 1234.93, 429],
    'Cd': [48, 112.41, 1.48, 1.69, 8.994, 12, 594.22, 96.6],
    'In': [49, 114.82, 1.44, 1.78, 5.786, 3, 429.75, 81.8],
    'Sn': [50, 118.71, 1.41, 1.96, 7.344, 4, 505.08, 66.8],
    'Sb': [51, 121.76, 1.38, 2.05, 8.64, 5, 903.78, 24.4],
    'Te': [52, 127.60, 1.35, 2.10, 9.010, 6, 722.66, 3.0],
    'I':  [53, 126.90, 1.33, 2.66, 10.451, 7, 386.85, 0.45],
    'Xe': [54, 131.29, 1.30, 2.60, 12.130, 0, 161.4, 0.006],
    'Cs': [55, 132.91, 2.25, 0.79, 3.894, 1, 301.59, 35.9],
    'Ba': [56, 137.33, 1.98, 0.89, 5.212, 2, 1000, 18.4],
    'La': [57, 138.91, 1.69, 1.10, 5.577, 3, 1193, 13.4],
    'Ce': [58, 140.12, 1.65, 1.12, 5.539, 4, 1068, 11.3],
    'Pr': [59, 140.91, 1.65, 1.13, 5.464, 3, 1208, 12.5],
    'Nd': [60, 144.24, 1.64, 1.14, 5.525, 3, 1297, 16.5],
    'Pm': [61, 145.0, 1.63, 1.13, 5.55, 3, 1315, 17.9],
    'Sm': [62, 150.36, 1.62, 1.17, 5.644, 3, 1345, 13.3],
    'Eu': [63, 151.96, 1.85, 1.20, 5.670, 2, 1099, 13.9],
    'Gd': [64, 157.25, 1.61, 1.20, 6.150, 3, 1585, 10.6],
    'Tb': [65, 158.93, 1.59, 1.10, 5.864, 3, 1629, 11.1],
    'Dy': [66, 162.50, 1.59, 1.22, 5.939, 3, 1680, 10.7],
    'Ho': [67, 164.93, 1.58, 1.23, 6.022, 3, 1734, 16.2],
    'Er': [68, 167.26, 1.57, 1.24, 6.108, 3, 1802, 14.5],
    'Tm': [69, 168.93, 1.56, 1.25, 6.184, 3, 1818, 16.9],
    'Yb': [70, 173.05, 1.74, 1.10, 6.254, 2, 1097, 38.5],
    'Lu': [71, 174.97, 1.56, 1.27, 5.426, 3, 1925, 16.4],
    'Hf': [72, 178.49, 1.44, 1.30, 6.825, 4, 2506, 23.0],
    'Ta': [73, 180.95, 1.34, 1.50, 7.89, 5, 3290, 57.5],
    'W':  [74, 183.84, 1.30, 2.36, 7.98, 6, 3695, 173],
    'Re': [75, 186.21, 1.28, 1.90, 7.88, 7, 3459, 48.0],
    'Os': [76, 190.23, 1.26, 2.20, 8.7, 8, 3306, 87.6],
    'Ir': [77, 192.22, 1.27, 2.20, 9.1, 9, 2719, 147],
    'Pt': [78, 195.08, 1.30, 2.28, 9.0, 10, 2041.4, 71.6],
    'Au': [79, 196.97, 1.34, 2.54, 9.226, 11, 1337.33, 318],
    'Hg': [80, 200.59, 1.49, 2.00, 10.438, 12, 234.32, 8.3],
    'Tl': [81, 204.38, 1.48, 1.62, 6.108, 3, 577, 46.1],
    'Pb': [82, 207.2, 1.47, 2.33, 7.417, 4, 600.61, 35.3],
    'Bi': [83, 208.98, 1.46, 2.02, 7.289, 5, 544.7, 7.97],
    'Po': [84, 209.0, 1.40, 2.00, 8.414, 6, 527, 20.0],
    'Th': [90, 232.04, 1.79, 1.30, 6.08, 4, 2023, 54.0],
    'Pa': [91, 231.04, 1.63, 1.50, 5.89, 5, 1841, 47.0],
    'U':  [92, 238.03, 1.56, 1.38, 6.194, 6, 1405.3, 27.5],
}

# 元素分类
ALKALI_METALS = {'Li', 'Na', 'K', 'Rb', 'Cs'}
ALKALINE_EARTH = {'Be', 'Mg', 'Ca', 'Sr', 'Ba'}
TRANSITION_METALS = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
}
LANTHANIDES = {'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
ACTINIDES = {'Th', 'Pa', 'U'}
HALOGENS = {'F', 'Cl', 'Br', 'I'}
CHALCOGENS = {'O', 'S', 'Se', 'Te'}
PNICTOGENS = {'N', 'P', 'As', 'Sb', 'Bi'}

# d0过渡金属 (常见于铁电材料)
D0_METALS = {'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Mo', 'W'}

# 常见氧化态
COMMON_OXIDATION_STATES = {
    'O': -2, 'S': -2, 'Se': -2, 'Te': -2,
    'F': -1, 'Cl': -1, 'Br': -1, 'I': -1,
    'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
    'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
    'Ti': 4, 'Zr': 4, 'Hf': 4,
    'V': 5, 'Nb': 5, 'Ta': 5,
    'Cr': 3, 'Mo': 6, 'W': 6,
    'Mn': 4, 'Fe': 3, 'Co': 3, 'Ni': 2,
    'Cu': 2, 'Zn': 2, 'Ag': 1, 'Cd': 2,
    'Al': 3, 'Ga': 3, 'In': 3,
    'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 2,
    'Bi': 3, 'La': 3, 'Y': 3, 'Th': 4, 'U': 6
}

# 极性点群
POLAR_POINT_GROUPS = ['1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm']

# 晶系映射
CRYSTAL_SYSTEM_MAP = {
    'triclinic': 0, 'monoclinic': 1, 'orthorhombic': 2,
    'tetragonal': 3, 'trigonal': 4, 'hexagonal': 5, 'cubic': 6
}

# 元素索引映射
ELEMENT_TO_IDX = {el: i for i, el in enumerate(ELEMENT_DATABASE.keys())}
IDX_TO_ELEMENT = {i: el for el, i in ELEMENT_TO_IDX.items()}
NUM_ELEMENTS = len(ELEMENT_DATABASE)

# ==========================================
# 2. 特征配置常量
# ==========================================
FEATURE_DIM = 64  # 总特征维度

# 特征分组
LATTICE_FEAT_DIM = 12       # 晶格参数特征
COMPOSITION_FEAT_DIM = 24   # 化学成分特征
SYMMETRY_FEAT_DIM = 8       # 对称性特征
BONDING_FEAT_DIM = 12       # 配位/键合特征
THERMO_FEAT_DIM = 8         # 热力学特征

# 输出目标配置
MAX_ELEMENTS = 5            # 最多5种元素
LATTICE_TARGET_DIM = 6      # vol_root, b/a, c/a, alpha, beta, gamma
COMPOSITION_TARGET_DIM = MAX_ELEMENTS * 2  # 元素索引 + 比例
SPACEGROUP_TARGET_DIM = 1
TOTAL_TARGET_DIM = LATTICE_TARGET_DIM + COMPOSITION_TARGET_DIM + SPACEGROUP_TARGET_DIM  # 17

# 特征名称 (用于报告)
FEATURE_NAMES = [
    # 晶格参数特征 [0-11]
    'Volume', 'Density', 'VolPerAtom', 
    'a_norm', 'b_norm', 'c_norm',
    'a/b', 'a/c', 'b/c',
    'alpha_dev', 'beta_dev', 'gamma_dev',
    # 化学成分特征 [12-35]
    'AvgMass', 'AvgRadius', 'AvgEN', 'AvgIE', 'AvgValence', 'AvgMeltingPt',
    'StdMass', 'StdRadius', 'StdEN', 'StdIE',
    'MaxRadius', 'MinRadius', 'RangeRadius', 'RatioRadius',
    'MaxEN', 'MinEN', 'RangeEN',
    'NumElements', 'TM_Frac', 'Lanthanide_Frac', 'D0_Frac', 'O_Frac', 'Halogen_Frac',
    # 对称性特征 [36-43]
    'SpaceGroup', 'CrystalSystem', 'IsPolar', 'PointGroupOrder',
    'HasInversion', 'NumSymOps', 'Multiplicity', 'WyckoffDiversity',
    # 配位/键合特征 [44-55]
    'AvgCoordNum', 'StdCoordNum', 'MaxCoordNum', 'MinCoordNum',
    'AvgBondLength', 'StdBondLength', 'MinBondLength', 'MaxBondLength',
    'BondLengthRange', 'AvgIonicChar', 'Tolerance_Factor', 'Goldschmidt',
    # 热力学特征 [56-63]
    'AvgThermalCond', 'StdThermalCond', 'FormationEnergy_Est', 'Cohesive_Est',
    'Electronegativity_Diff', 'Charge_Balance', 'Stability_Index', 'Polarizability_Est'
]


# ==========================================
# 3. 特征提取类
# ==========================================
class UnifiedFeatureExtractor:
    """
    统一特征提取器
    
    特征维度: 64维
    所有模型共享此特征工程
    """
    
    def __init__(self):
        self.feature_dim = FEATURE_DIM
        self.feature_names = FEATURE_NAMES
    
    @staticmethod
    def get_element_data(symbol: str) -> List[float]:
        """获取元素物理化学数据"""
        if symbol in ELEMENT_DATABASE:
            return ELEMENT_DATABASE[symbol]
        else:
            # 默认值
            return [50, 100, 1.5, 2.0, 7.0, 4, 1500, 50]
    
    @staticmethod
    def estimate_tolerance_factor(comp: Dict[str, float], radii: Dict[str, float]) -> float:
        """估算钙钛矿容忍因子 (Goldschmidt tolerance factor)"""
        # 简化版: 假设 ABO3 结构
        # t = (r_A + r_O) / (sqrt(2) * (r_B + r_O))
        try:
            elements = list(comp.keys())
            if 'O' not in elements or len(elements) < 3:
                return 1.0
            
            r_O = 1.40  # 氧离子半径
            
            # 找到可能的 A 位和 B 位元素
            non_O = [el for el in elements if el != 'O']
            if len(non_O) < 2:
                return 1.0
            
            # 按半径排序，大的是 A 位，小的是 B 位
            non_O_sorted = sorted(non_O, key=lambda x: radii.get(x, 1.5), reverse=True)
            r_A = radii.get(non_O_sorted[0], 1.5)
            r_B = radii.get(non_O_sorted[1], 1.0)
            
            t = (r_A + r_O) / (np.sqrt(2) * (r_B + r_O))
            return t
        except:
            return 1.0
    
    def extract_from_structure_dict(self, struct_dict: Dict[str, Any], 
                                     spacegroup_number: int = None,
                                     band_gap: float = None) -> np.ndarray:
        """
        从结构字典提取64维特征向量
        
        Args:
            struct_dict: pymatgen 结构的字典形式
            spacegroup_number: 空间群号 (可选)
            band_gap: 带隙 (可选)
        
        Returns:
            64维特征向量
        """
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        
        try:
            sites = struct_dict.get('sites', [])
            lattice = struct_dict.get('lattice', {})
            
            if not sites or not lattice:
                return features
            
            # ========== 晶格参数 ==========
            a = lattice.get('a', 1)
            b = lattice.get('b', 1)
            c = lattice.get('c', 1)
            alpha = lattice.get('alpha', 90)
            beta = lattice.get('beta', 90)
            gamma = lattice.get('gamma', 90)
            vol = lattice.get('volume', 1)
            
            if vol <= 0 or a <= 0:
                return features
            
            # ========== 成分分析 ==========
            comp = {}
            total_atoms = 0
            for site in sites:
                for species in site['species']:
                    el = species['element']
                    occu = species['occu']
                    comp[el] = comp.get(el, 0) + occu
                    total_atoms += occu
            
            if total_atoms == 0:
                return features
            
            # 归一化成分
            comp_frac = {el: count / total_atoms for el, count in comp.items()}
            
            # ========== 元素特征统计 ==========
            masses, radii, ens, ies, valences, melting_pts, thermal_conds = [], [], [], [], [], [], []
            radii_dict = {}
            
            tm_frac, lanthanide_frac, d0_frac, o_frac, halogen_frac = 0, 0, 0, 0, 0
            
            for el, frac in comp_frac.items():
                data = self.get_element_data(el)
                masses.append(data[1])
                radii.append(data[2])
                radii_dict[el] = data[2]
                ens.append(data[3])
                ies.append(data[4])
                valences.append(data[5])
                melting_pts.append(data[6])
                thermal_conds.append(data[7])
                
                if el in TRANSITION_METALS:
                    tm_frac += frac
                if el in LANTHANIDES:
                    lanthanide_frac += frac
                if el in D0_METALS:
                    d0_frac += frac
                if el == 'O':
                    o_frac += frac
                if el in HALOGENS:
                    halogen_frac += frac
            
            # 总质量用于密度计算
            total_mass = sum(data[1] * comp.get(el, 0) for el, data in 
                           [(el, self.get_element_data(el)) for el in comp.keys()])
            density = total_mass / vol if vol > 0 else 0
            
            # ========== 特征计算 ==========
            
            # [0-11] 晶格参数特征
            features[0] = min(vol, 5000) / 5000.0
            features[1] = min(density, 15) / 15.0
            features[2] = min(vol / total_atoms, 50) / 50.0
            features[3] = min(a, 30) / 30.0
            features[4] = min(b, 30) / 30.0
            features[5] = min(c, 30) / 30.0
            features[6] = min(a / max(b, 0.1), 3) / 3.0
            features[7] = min(a / max(c, 0.1), 3) / 3.0
            features[8] = min(b / max(c, 0.1), 3) / 3.0
            features[9] = (alpha - 90) / 90.0
            features[10] = (beta - 90) / 90.0
            features[11] = (gamma - 90) / 90.0
            
            # [12-35] 化学成分特征
            features[12] = np.mean(masses) / 200.0 if masses else 0
            features[13] = np.mean(radii) / 2.5 if radii else 0
            features[14] = np.mean(ens) / 4.0 if ens else 0
            features[15] = np.mean(ies) / 15.0 if ies else 0
            features[16] = np.mean(valences) / 8.0 if valences else 0
            features[17] = np.mean(melting_pts) / 3500.0 if melting_pts else 0
            
            features[18] = np.std(masses) / 100.0 if len(masses) > 1 else 0
            features[19] = np.std(radii) / 1.0 if len(radii) > 1 else 0
            features[20] = np.std(ens) / 2.0 if len(ens) > 1 else 0
            features[21] = np.std(ies) / 5.0 if len(ies) > 1 else 0
            
            features[22] = max(radii) / 2.5 if radii else 0
            features[23] = min(radii) / 2.5 if radii else 0
            features[24] = (max(radii) - min(radii)) / 2.0 if radii else 0
            features[25] = max(radii) / max(min(radii), 0.1) if radii else 1
            features[25] = min(features[25], 5) / 5.0
            
            features[26] = max(ens) / 4.0 if ens else 0
            features[27] = min(ens) / 4.0 if ens else 0
            features[28] = (max(ens) - min(ens)) / 4.0 if ens else 0
            
            features[29] = len(comp) / 10.0
            features[30] = tm_frac
            features[31] = lanthanide_frac
            features[32] = d0_frac
            features[33] = o_frac
            features[34] = halogen_frac
            features[35] = 0  # 保留位
            
            # [36-43] 对称性特征
            sg = spacegroup_number if spacegroup_number else 1
            features[36] = sg / 230.0
            
            # 根据空间群号估算晶系
            if sg <= 2:
                crystal_sys = 0  # triclinic
            elif sg <= 15:
                crystal_sys = 1  # monoclinic
            elif sg <= 74:
                crystal_sys = 2  # orthorhombic
            elif sg <= 142:
                crystal_sys = 3  # tetragonal
            elif sg <= 167:
                crystal_sys = 4  # trigonal
            elif sg <= 194:
                crystal_sys = 5  # hexagonal
            else:
                crystal_sys = 6  # cubic
            
            features[37] = crystal_sys / 6.0
            
            # 极性估计 (基于常见的铁电空间群)
            polar_spacegroups = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                 75, 76, 77, 78, 79, 80, 99, 100, 101, 102, 103, 104, 105, 106,
                                 143, 144, 145, 146, 156, 157, 158, 159, 160, 161,
                                 168, 169, 170, 171, 172, 173, 183, 184, 185, 186}
            features[38] = 1.0 if sg in polar_spacegroups else 0.0
            
            features[39] = 0.5  # 点群阶数估计
            features[40] = 0.0 if sg in polar_spacegroups else 1.0  # 反演对称
            features[41] = min(sg, 230) / 230.0  # 对称操作数估计
            features[42] = total_atoms / 50.0  # 多重度
            features[43] = len(comp) / 10.0  # Wyckoff多样性
            
            # [44-55] 配位/键合特征 (估算)
            # 使用简单的几何估计
            avg_bond_est = np.mean(radii) * 2 if radii else 2.0
            features[44] = 6.0 / 12.0  # 假设平均配位数 6
            features[45] = 0.2  # 配位数标准差
            features[46] = 12.0 / 12.0  # 最大配位数
            features[47] = 4.0 / 12.0   # 最小配位数
            features[48] = min(avg_bond_est, 4) / 4.0
            features[49] = 0.2  # 键长标准差
            features[50] = min(avg_bond_est * 0.8, 4) / 4.0
            features[51] = min(avg_bond_est * 1.2, 4) / 4.0
            features[52] = 0.2  # 键长范围
            
            # 离子性估计 (电负性差)
            if len(ens) > 1:
                ionic_char = (max(ens) - min(ens)) / 3.0
            else:
                ionic_char = 0
            features[53] = min(ionic_char, 1.0)
            
            # 容忍因子
            t_factor = self.estimate_tolerance_factor(comp_frac, radii_dict)
            features[54] = min(abs(t_factor), 2) / 2.0
            features[55] = min(abs(1 - t_factor), 1)  # 偏离1的程度
            
            # [56-63] 热力学特征
            features[56] = np.mean(thermal_conds) / 200.0 if thermal_conds else 0
            features[56] = min(features[56], 1.0)
            features[57] = np.std(thermal_conds) / 100.0 if len(thermal_conds) > 1 else 0
            features[57] = min(features[57], 1.0)
            
            # 形成能估计 (基于电负性差和离子性)
            features[58] = ionic_char * 0.5
            
            # 内聚能估计
            features[59] = np.mean(melting_pts) / 3500.0 if melting_pts else 0
            
            # 电负性差
            features[60] = (max(ens) - min(ens)) / 4.0 if len(ens) > 1 else 0
            
            # 电荷平衡估计
            total_charge = 0
            for el, frac in comp_frac.items():
                if el in COMMON_OXIDATION_STATES:
                    total_charge += COMMON_OXIDATION_STATES[el] * frac * total_atoms
            features[61] = 1.0 - min(abs(total_charge), 10) / 10.0
            
            # 稳定性指数
            stability = features[61] * (1 - abs(1 - t_factor))
            features[62] = min(max(stability, 0), 1)
            
            # 极化率估计 (基于原子半径)
            features[63] = np.mean([r**3 for r in radii]) / 10.0 if radii else 0
            features[63] = min(features[63], 1.0)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
        
        return features
    
    def extract_target_vector(self, struct_dict: Dict[str, Any], 
                              spacegroup_number: int = None) -> Optional[np.ndarray]:
        """
        提取目标向量 (用于逆向设计训练)
        
        目标维度: 17
        - 晶格: vol_root, b/a, c/a, alpha, beta, gamma (6维)
        - 成分: 5元素 × (索引 + 比例) (10维)
        - 空间群: 1维
        """
        try:
            lattice = struct_dict.get('lattice', {})
            sites = struct_dict.get('sites', [])
            
            a = lattice.get('a', 0)
            b = lattice.get('b', 0)
            c = lattice.get('c', 0)
            alpha = lattice.get('alpha', 90)
            beta = lattice.get('beta', 90)
            gamma = lattice.get('gamma', 90)
            vol = lattice.get('volume', 0)
            
            if vol <= 0 or a <= 0:
                return None
            
            # 晶格目标
            vol_root = vol ** (1/3)
            ratio_ba = b / a
            ratio_ca = c / a
            
            lattice_targets = [vol_root, ratio_ba, ratio_ca, alpha, beta, gamma]
            
            # 成分
            comp = {}
            total_atoms = 0
            for site in sites:
                for species in site['species']:
                    el = species['element']
                    occu = species['occu']
                    comp[el] = comp.get(el, 0) + occu
                    total_atoms += occu
            
            # 编码成分
            el_list = []
            for el, count in comp.items():
                if el in ELEMENT_TO_IDX:
                    idx = ELEMENT_TO_IDX[el]
                    frac = count / total_atoms
                    el_list.append((idx, frac))
            
            el_list.sort(key=lambda x: x[0])
            
            comp_vector = []
            for i in range(MAX_ELEMENTS):
                if i < len(el_list):
                    comp_vector.extend([el_list[i][0] / NUM_ELEMENTS, el_list[i][1]])
                else:
                    comp_vector.extend([0, 0])
            
            # 空间群
            sg = spacegroup_number if spacegroup_number else 1
            sg_norm = sg / 230.0
            
            target = lattice_targets + comp_vector + [sg_norm]
            return np.array(target, dtype=np.float32)
            
        except Exception as e:
            return None


# ==========================================
# 4. 便捷函数
# ==========================================
_extractor = None

def get_extractor() -> UnifiedFeatureExtractor:
    """获取全局特征提取器实例"""
    global _extractor
    if _extractor is None:
        _extractor = UnifiedFeatureExtractor()
    return _extractor


def extract_features(struct_dict: Dict[str, Any], 
                     spacegroup_number: int = None) -> np.ndarray:
    """便捷函数: 提取特征向量"""
    return get_extractor().extract_from_structure_dict(struct_dict, spacegroup_number)


def extract_target(struct_dict: Dict[str, Any], 
                   spacegroup_number: int = None) -> Optional[np.ndarray]:
    """便捷函数: 提取目标向量"""
    return get_extractor().extract_target_vector(struct_dict, spacegroup_number)


def decode_composition(comp_vector: np.ndarray) -> List[Dict[str, Any]]:
    """
    解码成分向量为元素列表
    
    Args:
        comp_vector: 长度为 MAX_ELEMENTS*2 的向量
    
    Returns:
        元素列表 [{'element': 'Ba', 'fraction': 0.2}, ...]
    """
    elements = []
    for i in range(MAX_ELEMENTS):
        idx_norm = comp_vector[i * 2]
        frac = comp_vector[i * 2 + 1]
        
        if frac > 0.02:  # 过滤小于2%的元素
            idx = int(idx_norm * NUM_ELEMENTS)
            idx = max(0, min(idx, NUM_ELEMENTS - 1))
            el = IDX_TO_ELEMENT.get(idx, 'X')
            elements.append({'element': el, 'fraction': float(frac)})
    
    # 归一化
    total = sum(e['fraction'] for e in elements)
    if total > 0:
        for e in elements:
            e['fraction'] /= total
    
    return elements


def decode_lattice(lattice_vector: np.ndarray) -> Dict[str, float]:
    """
    解码晶格向量为晶格参数
    
    Args:
        lattice_vector: [vol_root, b/a, c/a, alpha, beta, gamma]
    
    Returns:
        {'a': ..., 'b': ..., 'c': ..., 'alpha': ..., 'beta': ..., 'gamma': ..., 'volume': ...}
    """
    vol_root = lattice_vector[0]
    ratio_ba = max(lattice_vector[1], 0.1)
    ratio_ca = max(lattice_vector[2], 0.1)
    alpha = lattice_vector[3]
    beta = lattice_vector[4]
    gamma = lattice_vector[5]
    
    # 还原 a, b, c
    a = vol_root / (ratio_ba * ratio_ca) ** (1/3)
    b = a * ratio_ba
    c = a * ratio_ca
    vol = vol_root ** 3
    
    return {
        'a': float(a),
        'b': float(b),
        'c': float(c),
        'alpha': float(alpha),
        'beta': float(beta),
        'gamma': float(gamma),
        'volume': float(vol)
    }


# ==========================================
# 5. 测试函数
# ==========================================
def test_feature_extraction():
    """测试特征提取"""
    import json
    
    print("Testing Unified Feature Extraction...")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Target dimension: {TOTAL_TARGET_DIM}")
    
    # 测试数据
    test_file = 'new_data/dataset_original_ferroelectric.jsonl'
    
    try:
        with open(test_file, 'r') as f:
            item = json.loads(f.readline())
        
        extractor = UnifiedFeatureExtractor()
        
        # 提取特征
        features = extractor.extract_from_structure_dict(
            item['structure'], 
            item.get('spacegroup_number')
        )
        
        print(f"\nFeature vector shape: {features.shape}")
        print(f"Non-zero features: {np.count_nonzero(features)}")
        
        # 显示部分特征
        print("\nSample features:")
        for i in range(0, min(20, FEATURE_DIM)):
            print(f"  {FEATURE_NAMES[i]}: {features[i]:.4f}")
        
        # 提取目标
        target = extractor.extract_target_vector(
            item['structure'], 
            item.get('spacegroup_number')
        )
        
        if target is not None:
            print(f"\nTarget vector shape: {target.shape}")
            print(f"Lattice targets: {target[:6]}")
            
            # 解码成分
            comp = decode_composition(target[6:16])
            print(f"Decoded composition: {comp}")
            
            # 解码晶格
            lattice = decode_lattice(target[:6])
            print(f"Decoded lattice: a={lattice['a']:.3f}, b={lattice['b']:.3f}, c={lattice['c']:.3f}")
        
        print("\n✓ Feature extraction test passed!")
        
    except FileNotFoundError:
        print(f"Test file not found: {test_file}")
    except Exception as e:
        print(f"Test error: {e}")


if __name__ == '__main__':
    test_feature_extraction()
