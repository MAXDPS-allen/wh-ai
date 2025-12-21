import pandas as pd
import numpy as np
import json
import random
import joblib  # <--- 核心修改：使用 joblib 替代 pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略版本兼容性警告，避免刷屏
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. 基础配置与特征计算逻辑 (保持不变)
# ==========================================
element_data = {
    'H': [1, 1.008, 0.37, 2.20, 13.598, 1], 'He': [2, 4.0026, 0.32, 0.0, 24.587, 8], 
    'Li': [3, 6.94, 1.34, 0.98, 5.392, 1], 'Be': [4, 9.0122, 0.90, 1.57, 9.323, 2],
    'B': [5, 10.81, 0.82, 2.04, 8.298, 3], 'C': [6, 12.011, 0.77, 2.55, 11.260, 4], 
    'N': [7, 14.007, 0.75, 3.04, 14.534, 5], 'O': [8, 15.999, 0.73, 3.44, 13.618, 6],
    'F': [9, 18.998, 0.71, 3.98, 17.423, 7], 'Ne': [10, 20.180, 0.69, 0.0, 21.565, 8], 
    'Na': [11, 22.990, 1.54, 0.93, 5.139, 1], 'Mg': [12, 24.305, 1.30, 1.31, 7.646, 2],
    'Al': [13, 26.982, 1.18, 1.61, 5.986, 3], 'Si': [14, 28.085, 1.11, 1.90, 8.152, 4], 
    'P': [15, 30.974, 1.06, 2.19, 10.487, 5], 'S': [16, 32.06, 1.02, 2.58, 10.360, 6],
    'Cl': [17, 35.45, 0.99, 3.16, 12.968, 7], 'K': [19, 39.098, 1.96, 0.82, 4.341, 1], 
    'Ca': [20, 40.078, 1.74, 1.00, 6.113, 2], 'Sc': [21, 44.956, 1.44, 1.36, 6.561, 3],
    'Ti': [22, 47.867, 1.36, 1.54, 6.828, 4], 'V': [23, 50.942, 1.25, 1.63, 6.746, 5], 
    'Cr': [24, 51.996, 1.27, 1.66, 6.767, 6], 'Mn': [25, 54.938, 1.39, 1.55, 7.434, 7],
    'Fe': [26, 55.845, 1.25, 1.83, 7.902, 8], 'Co': [27, 58.933, 1.26, 1.88, 7.881, 9], 
    'Ni': [28, 58.693, 1.21, 1.91, 7.640, 10], 'Cu': [29, 63.546, 1.38, 1.90, 7.726, 11],
    'Zn': [30, 65.38, 1.31, 1.65, 9.394, 12], 'Ga': [31, 69.723, 1.26, 1.81, 6.000, 3], 
    'Ge': [32, 72.63, 1.22, 2.01, 7.900, 4], 'As': [33, 74.922, 1.19, 2.18, 9.789, 5],
    'Se': [34, 78.96, 1.16, 2.55, 9.752, 6], 'Br': [35, 79.904, 1.14, 2.96, 11.814, 7], 
    'Rb': [37, 85.468, 2.11, 0.82, 4.177, 1], 'Sr': [38, 87.62, 1.92, 0.95, 5.695, 2],
    'Y': [39, 88.906, 1.62, 1.22, 6.217, 3], 'Zr': [40, 91.224, 1.48, 1.33, 6.634, 4], 
    'Nb': [41, 92.906, 1.37, 1.60, 6.759, 5], 'Mo': [42, 95.96, 1.45, 2.16, 7.092, 6],
    'Tc': [43, 98.0, 1.56, 1.90, 7.28, 7], 'Ru': [44, 101.07, 1.26, 2.20, 7.361, 8], 
    'Rh': [45, 102.91, 1.35, 2.28, 7.459, 9], 'Pd': [46, 106.42, 1.31, 2.20, 8.337, 10],
    'Ag': [47, 107.87, 1.53, 1.93, 7.576, 11], 'Cd': [48, 112.41, 1.48, 1.69, 8.994, 12], 
    'In': [49, 114.82, 1.44, 1.78, 5.786, 3], 'Sn': [50, 118.71, 1.41, 1.96, 7.344, 4],
    'Sb': [51, 121.76, 1.38, 2.05, 8.64, 5], 'Te': [52, 127.60, 1.35, 2.10, 9.010, 6], 
    'I': [53, 126.90, 1.33, 2.66, 10.451, 7], 'Cs': [55, 132.91, 2.25, 0.79, 3.894, 1],
    'Ba': [56, 137.33, 1.98, 0.89, 5.212, 2], 'La': [57, 138.91, 1.69, 1.10, 5.577, 3], 
    'Ce': [58, 140.12, 1.65, 1.12, 5.539, 3], 'Pr': [59, 140.91, 1.65, 1.13, 5.464, 3],
    'Nd': [60, 144.24, 1.64, 1.14, 5.525, 3], 'Pm': [61, 145.0, 1.63, 1.13, 5.55, 3], 
    'Sm': [62, 150.36, 1.62, 1.17, 5.644, 3], 'Eu': [63, 151.96, 1.85, 1.20, 5.670, 2],
    'Gd': [64, 157.25, 1.61, 1.20, 6.150, 3], 'Tb': [65, 158.93, 1.59, 1.10, 5.864, 3], 
    'Dy': [66, 162.50, 1.59, 1.22, 5.939, 3], 'Ho': [67, 164.93, 1.58, 1.23, 6.022, 3],
    'Er': [68, 167.26, 1.57, 1.24, 6.108, 3], 'Tm': [69, 168.93, 1.56, 1.25, 6.184, 3], 
    'Yb': [70, 173.05, 1.74, 1.10, 6.254, 2], 'Lu': [71, 174.97, 1.56, 1.27, 5.426, 3],
    'Hf': [72, 178.49, 1.44, 1.30, 6.825, 4], 'Ta': [73, 180.95, 1.34, 1.50, 7.89, 5], 
    'W': [74, 183.84, 1.30, 2.36, 7.98, 6], 'Re': [75, 186.21, 1.28, 1.90, 7.88, 7],
    'Os': [76, 190.23, 1.26, 2.20, 8.7, 8], 'Ir': [77, 192.22, 1.27, 2.20, 9.1, 9], 
    'Pt': [78, 195.08, 1.30, 2.28, 9.0, 10], 'Au': [79, 196.97, 1.34, 2.54, 9.226, 11],
    'Hg': [80, 200.59, 1.49, 2.00, 10.438, 12], 'Tl': [81, 204.38, 1.48, 1.62, 6.108, 3], 
    'Pb': [82, 207.2, 1.47, 2.33, 7.417, 4], 'Bi': [83, 208.98, 1.46, 2.02, 7.289, 5],
    'Th': [90, 232.04, 1.79, 1.30, 6.08, 4], 'Pa': [91, 231.04, 1.63, 1.50, 5.89, 5], 
    'U': [92, 238.03, 1.56, 1.38, 6.194, 6]
}
TM_SET = set(['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'])

def compute_features(item):
    if 'structure' not in item: return None
    struct = item['structure']
    sites = struct.get('sites', [])
    lattice = struct.get('lattice', {})
    
    comp = {}
    total_atoms = 0
    for site in sites:
        for species in site['species']:
            el = species['element']
            occu = species['occu']
            comp[el] = comp.get(el, 0) + occu
            total_atoms += occu
            
    if total_atoms == 0: return None
    
    props = [[] for _ in range(5)]
    tm_fraction = 0.0
    current_mass_sum = 0
    
    for el, count in comp.items():
        if el in element_data:
            data = element_data[el]
            frac = count / total_atoms
            current_mass_sum += data[1] * count
            for i in range(5):
                props[i].append((data[i+1], frac))
            if el in TM_SET: tm_fraction += frac

    if not props[0]: return None
    
    feats = []
    # 1. Chemistry Avg
    for i in range(5):
        vals = [p[0] for p in props[i]]
        weights = [p[1] for p in props[i]]
        feats.append(np.average(vals, weights=weights))
        
    # 2. Radius Stats
    radii = [p[0] for p in props[1]]
    max_r, min_r = max(radii), min(radii)
    feats.extend([max_r, min_r, max_r - min_r])
    feats.append(max_r / min_r if min_r > 0 else 0)
    
    # 3. EN Stats
    ens = [p[0] for p in props[2]]
    max_en, min_en = max(ens), min(ens)
    feats.extend([max_en - min_en])
    
    # 4. Structure Stats
    a, b, c = lattice.get('a', 0), lattice.get('b', 0), lattice.get('c', 0)
    alpha, beta, gamma = lattice.get('alpha', 90), lattice.get('beta', 90), lattice.get('gamma', 90)
    vol = lattice.get('volume', 0)
    
    angle_distortion = abs(alpha - 90) + abs(beta - 90) + abs(gamma - 90)
    feats.append(angle_distortion)
    
    lengths = [x for x in [a, b, c] if x > 0]
    feats.append(max(lengths) / min(lengths) if lengths else 1.0)
    
    density = current_mass_sum / vol if vol > 0 else 0
    feats.append(density)
    
    vol_per_atom = vol / total_atoms if total_atoms > 0 else 0
    feats.append(vol_per_atom)
    
    sg = item.get('spacegroup_number', 0)
    if sg is None: sg = 0
    feats.append(sg)

    feats.append(tm_fraction)
    feats.append(len(comp))
    bg = item.get('band_gap')
    if bg is None: bg = 0
    feats.append(bg)
        
    return feats

# ==========================================
# 2. 加载预训练模型 (修改部分)
# ==========================================
print("Loading pre-trained model files with joblib...")

try:
    with open('model_v3/v_3_fe_features.json', 'r') as f:
        FEATURE_NAMES = json.load(f)
    print(f"Features loaded: {len(FEATURE_NAMES)} features")
except Exception as e:
    print(f"Error loading features json: {e}")
    exit()

try:
    # 修改点：直接使用 joblib.load
    scaler = joblib.load('model_v3/v_3_fe_scaler.pkl')
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    # 尝试备用方案：如果是 pickle 保存的 joblib 对象
    try:
        import pickle
        with open('model_v3/v_3_fe_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded with pickle fallback.")
    except:
        exit()

try:
    # 模型也建议用 joblib 加载
    clf = joblib.load('model_v3/v_3_fe_model.pkl')
    print("XGBoost Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ==========================================
# 3. 准备种子数据 (Seeding)
# ==========================================
pos_files = ['new_data/dataset_original_ferroelectric.jsonl', 'new_data/dataset_known_FE_rest.jsonl']
X_pos_raw = []

print("Extracting seed features from positive datasets...")
for f_name in pos_files:
    try:
        with open(f_name, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    feats = compute_features(item)
                    if feats:
                        X_pos_raw.append(feats)
                except: continue
    except: pass

if len(X_pos_raw) == 0:
    print("Warning: No seed data found. Using random seeds.")
    X_pos_raw = np.random.rand(100, len(FEATURE_NAMES))
else:
    print(f"Loaded {len(X_pos_raw)} seed samples.")

X_pos_raw = np.array(X_pos_raw)

# ==========================================
# 4. 遗传算法生成器 (Genetic Algorithm)
# ==========================================

class GeneticGenerator:
    def __init__(self, classifier, scaler, seed_population, population_size=200):
        self.clf = classifier
        self.scaler = scaler
        self.pop_size = population_size
        self.feature_count = seed_population.shape[1]
        self.population = self._initialize_population(seed_population)

    def _initialize_population(self, seeds):
        indices = np.random.choice(len(seeds), self.pop_size, replace=True)
        base_pop = seeds[indices]
        scaled_pop = self.scaler.transform(base_pop)
        noise = np.random.normal(0, 0.2, scaled_pop.shape)
        return scaled_pop + noise

    def fitness(self, pop):
        # 使用加载的模型进行预测
        probs = self.clf.predict_proba(pop)[:, 1]
        return probs

    def select(self, pop, scores, keep_k=50):
        indices = np.argsort(scores)[::-1]
        top_indices = indices[:keep_k]
        return pop[top_indices]

    def crossover(self, parents, num_offspring):
        offspring = []
        for _ in range(num_offspring):
            p1, p2 = random.sample(list(parents), 2)
            mask = np.random.rand(self.feature_count) > 0.5
            child = np.where(mask, p1, p2)
            offspring.append(child)
        return np.array(offspring)

    def mutate(self, pop, mutation_rate=0.1, mutation_scale=0.1):
        mask = np.random.rand(*pop.shape) < mutation_rate
        noise = np.random.normal(0, mutation_scale, pop.shape)
        pop[mask] += noise[mask]
        return pop

    def evolve(self, generations=30):
        print(f"Starting Evolution using loaded XGBoost model...")
        for g in range(generations):
            scores = self.fitness(self.population)
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            
            if g % 10 == 0:
                print(f"Gen {g}: Max Prob={max_score:.4f}, Avg Prob={avg_score:.4f}")
            
            parents = self.select(self.population, scores, keep_k=self.pop_size // 4)
            num_children = self.pop_size - len(parents)
            children = self.crossover(parents, num_children)
            children = self.mutate(children)
            self.population = np.vstack([parents, children])
            
        final_scores = self.fitness(self.population)
        best_indices = np.argsort(final_scores)[::-1]
        
        # 逆标准化
        best_pop_scaled = self.population[best_indices]
        best_pop_original = self.scaler.inverse_transform(best_pop_scaled)
        
        return best_pop_original, final_scores[best_indices]

# 运行 GA
ga = GeneticGenerator(clf, scaler, X_pos_raw)
best_candidates, best_scores = ga.evolve(generations=50)

# ==========================================
# 5. 结果整理与保存
# ==========================================
mask = best_scores > 0.90
final_candidates = best_candidates[mask]
final_scores = best_scores[mask]

df_gen = pd.DataFrame(final_candidates, columns=FEATURE_NAMES)
df_gen['Predicted_FE_Prob'] = final_scores

# 物理约束修正
if 'SpaceGroup_Number' in df_gen.columns:
    df_gen['SpaceGroup_Number'] = df_gen['SpaceGroup_Number'].round().astype(int).clip(1, 230)
if 'Num_Elements' in df_gen.columns:
    df_gen['Num_Elements'] = df_gen['Num_Elements'].round().astype(int).clip(2, 8)

df_gen = df_gen.drop_duplicates(subset=FEATURE_NAMES)

print(f"\nGenerated {len(df_gen)} high-confidence candidates.")
if len(df_gen) > 0:
    print(df_gen.head())
    df_gen.to_csv('generated_candidates_v3.csv', index=False)
    print("Results saved to 'generated_candidates_v3.csv'")