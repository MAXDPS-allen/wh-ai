# 铁电材料分类模型完整总结报告
## Comprehensive Model Summary for Ferroelectric Material Classification

**生成时间**: 2026年1月 | **目标**: Accuracy ≥ 99%, Recall ≥ 99%

---

## 📊 目录

1. [项目概述](#1-项目概述)
2. [数据集](#2-数据集)
3. [代码版本演进](#3-代码版本演进)
4. [模型详细列表](#4-模型详细列表)
5. [性能对比表](#5-性能对比表)
6. [关键发现与经验](#6-关键发现与经验)
7. [推荐方案](#7-推荐方案)

---

## 1. 项目概述

### 1.1 研究目标
开发高精度铁电材料分类器，实现:
- **准确率 (Accuracy)**: ≥ 99%
- **召回率 (Recall)**: ≥ 99% (不漏掉任何铁电材料)
- **精确度 (Precision)**: ≥ 95%

### 1.2 技术挑战
| 挑战 | 描述 | 解决方案 |
|------|------|----------|
| 极端类别不平衡 | FE:Non-FE ≈ 1:36.3 | SMOTE, 类别权重, 过采样 |
| 结构复杂性 | 晶体结构数据多样 | 图神经网络, E(3)等变网络 |
| Recall与Precision平衡 | 提高Recall往往降低Precision | 代价敏感学习, 级联分类器 |

---

## 2. 数据集

### 2.1 数据集版本

| 数据集 | 正样本(FE) | 负样本(Non-FE) | 不平衡比 | 描述 |
|--------|-----------|---------------|---------|------|
| **原始数据集** | 156 | ~400 | 1:2.5 | ferroelectric_database_labeled.csv |
| **扩展数据集** | 664 | 24,130 | 1:36.3 | 完整24,794样本 |
| **极性材料子集** | 664 | 15,703 | 1:23.6 | 仅保留非中心对称材料 (16,367样本) |

### 2.2 数据文件
- `dataset_original_ferroelectric.jsonl`: 156个原始正样本
- `dataset_known_FE_rest.jsonl`: 508个扩展正样本
- `dataset_nonFE.jsonl`: 5,000负样本
- `dataset_nonFE_cleaned.jsonl`: 清洗后负样本
- `dataset_mp_polar.jsonl`: Materials Project极性材料

---

## 3. 代码版本演进

### 3.1 code_v1 - 传统机器学习基线
**时间**: 项目初期
**数据**: 原始小数据集 (~500样本)
**特点**: 手工特征工程 + 传统ML

| 文件 | 模型 | 特点 |
|------|------|------|
| `R_F_Feature.py` | Random Forest | 网格搜索超参数, 特征重要性分析 |
| `R_F_HGB_SVM_vot.py` | RF + HistGradientBoosting + SVM | Soft Voting集成 |
| `R_F_T_SVM_vot.py` | RF + GradientBoosting + SVM | 可调阈值Voting |

**特征工程**:
```
1. Distortion_Density = Max_Distortion / Volume_Change_Ratio
2. Bandgap_Strain_Interaction = Bandgap × Spontaneous_Strain
3. Symmetry_Distortion_Product = Symmetry_Change × Avg_Distortion
```

### 3.2 code_v2 - 深度学习探索
**时间**: 中期
**数据**: 扩展数据集
**特点**: 引入神经网络, 处理类别不平衡

| 文件 | 模型 | 特点 |
|------|------|------|
| `nd_NN.py` | MLP | 基础神经网络 |
| `nnd_NN_smote.py` | MLP + SMOTE | 自实现SMOTE过采样 |
| `nnd_XGBoost.py` | XGBoost | 梯度提升树 |
| `nnd_XGBoost_ev.py` | XGBoost增强版 | 添加物理先验 |
| `nnd_XGBoost_woHE.py` | XGBoost无超参编码 | 简化版 |
| `new_data_RF_HGB_SVM_vot.py` | RF+HGB+SVM | 在新数据上重新训练 |
| `nd_cho_vot.py` | Voting | 选择性投票 |
| `nd_vot.py` | Voting | 通用投票 |
| `v_2_XGb.py` | XGBoost v2 | 改进版 |

**物理知识嵌入**:
- 中心对称空间群列表 (92个)
- 关键铁电元素: O, Ti, Ba, Pb, Sr, Zr, Hf, Nb, Ta, Bi, K, Li
- 过渡金属集合判断

### 3.3 code_v3 - XGBoost优化
**时间**: 中后期
**特点**: 专注XGBoost调优

| 文件 | 描述 |
|------|------|
| `v3_XGB_save.py` | 模型保存功能 |
| `nnd_XGBoost_woHE.py` | 无超参数编码版本 |

### 3.4 code_v5 - 图神经网络时代
**时间**: 后期
**特点**: GNN架构探索, 生成模型

| 文件 | 模型类型 | 关键特性 |
|------|----------|----------|
| `GCNN.py` | GCN + GAT | 基础图卷积, 全局特征, 混合精度 |
| `GCNN_v2.py` | GCNN增强 | 64维统一特征 |
| `GCNN_v3.py` | GCNN高召回 | Focal Loss (α=0.85, γ=1.5), 最优阈值0.41 |
| `GCNN_v4.py` | GCNN极端召回 | 正样本权重=15, 阈值=0.07 |
| `GCNN_v5.py` | GCNN集成 | 3模型集成, 正样本权重=20, 过采样=3× |
| `GCNN_v6.py` | GCNN扩展 | 扩展负样本训练 |
| `NequIP_Classifier.py` | NequIP v1 | E(3)等变, Bessel径向基, 球谐函数 |
| `NequIP_Classifier_v2.py` | NequIP v2 | 改进版 |
| `FE_GAN.py` | GAN v1 | 生成铁电材料特征 |
| `FE_GAN_v2.py` | GAN v2 | 64维特征 |
| `FE_GAN_v3.py` | GAN v3 | t-SNE可视化 |
| `FE_GAN_v4.py` | GAN v4 | 进一步优化 |
| `FE_CVAE.py` | CVAE | 条件变分自编码器 |
| `screen_database.py` | 筛选工具 | 数据库筛选 |
| `screen_database_nequip.py` | NequIP筛选 | 使用NequIP模型筛选 |
| `full_discovery_pipeline.py` | 完整流程 | 端到端发现管道 |
| `generate_materials_pipeline.py` | 生成流程 | 材料生成管道 |

### 3.5 code_v6 - 高级优化
**时间**: 最近
**特点**: NequIP深度优化, 集成策略, 代价敏感学习

| 文件 | 模型 | 关键特性 |
|------|------|----------|
| `train_nequip_v6.py` | NequIP v6 | 基础极性材料训练 |
| `train_nequip_v7.log` | NequIP v7 | 6层SE(3)注意力, l=0,1,2,3 |
| `train_nequip_v8.log` | NequIP v8 | KDTree加速, 4层消息传递, 16维Bessel |
| `train_nequip_v9.py` | NequIP v9 | 10层ResNet, CrossAttention, CosineAnnealing |
| `train_nequip_v9_full.py` | NequIP v9完整 | 全数据训练 (Loss=nan失败) |
| `train_composite_v2.py` | Composite V2 | 4子模型: GCNN+NequIP+MLP+ResNet |
| `train_cascade.py` | Cascade | 两阶段: 高召回→高精确 |
| `train_ultimate.py` | Ultimate v1 | 集成分类器 |
| `train_ultimate_v2.py` | Ultimate v2 | 增强集成 |
| `train_ensemble_cost_sensitive.py` | Ensemble+CS | 3模型+代价敏感Focal Loss |
| `train_high_recall.py` | 高召回 | 专注召回率 |
| `train_transformer.py` | Transformer | 注意力机制 |
| `train_graph_nn.py` | Graph NN | 图神经网络 |
| `train_balanced.py` | 平衡训练 | 类别平衡策略 |

---

## 4. 模型详细列表

### 4.1 传统机器学习模型

#### RF + HistGradientBoosting + SVM Voting (code_v1)
- **架构**: 3个基模型Soft Voting
- **特点**: 
  - Random Forest: n_estimators=200
  - HistGradientBoosting: learning_rate=0.1, max_iter=100
  - SVM: RBF核, 标准化预处理
- **性能**: 原始小数据集上表现良好
- **状态**: ✅ 完成

#### XGBoost系列 (code_v2, v3)
- **版本**: 基础版, 增强版, 无超参编码版
- **特点**: 物理知识嵌入 (中心对称判定)
- **性能**: 对极端不平衡敏感
- **状态**: ✅ 完成

---

### 4.2 图神经网络模型

#### GCNN v2 (code_v5)
- **架构**: GATConv + 全局池化
- **特征**: 64维统一特征
- **训练**: 47 epochs
- **结果**:
  - Accuracy: 95.6%
  - AUC: 0.9845
- **状态**: ✅ 完成

#### GCNN v3 - 高召回优化 (code_v5)
- **架构**: GAT + Focal Loss
- **配置**:
  - Focal Loss: α=0.85, γ=1.5
  - 正样本权重: 10.0
  - 最优阈值: 0.41
- **训练**: 113 epochs
- **结果**:
  - Accuracy: 95.1%
  - Recall: 81.8% (最佳95.6%)
  - Precision: 77.2%
  - AUC: 0.9630
- **状态**: ✅ 完成

#### GCNN v4 - 极端召回 (code_v5)
- **配置**:
  - 正样本权重: 15.0
  - 最优阈值: 0.07
- **结果**:
  - Recall: 98.2%
  - Precision: 56.4%
  - AUC: 0.9875
- **状态**: ✅ 完成

#### GCNN v5 - 集成模型 (code_v5)
- **架构**: 3模型集成
- **配置**:
  - 正样本权重: 20.0
  - 过采样比例: 3.0
  - 最优阈值: 0.87
- **结果**:
  - Recall: 99.0%
  - Precision: 96.9%
  - AUC: 0.9972
- **状态**: ✅ 完成 ⭐ **最佳GCNN**

---

### 4.3 E(3)等变神经网络 (NequIP系列)

#### NequIP v1 (code_v5)
- **架构**: 
  - Bessel径向基 (8个)
  - 球谐函数 (l=0,1,2)
  - 4层消息传递
  - 截断半径: 5.0Å
- **结果**:
  - Accuracy: 97.84%
  - Recall: 96.60%
  - Precision: 95.90%
  - AUC: 0.9961
- **状态**: ✅ 完成

#### NequIP v6 (code_v6)
- **数据**: 极性材料子集
- **结果**:
  - Accuracy: 98.24%
  - Recall: 43.22%
  - AUC: 0.9377
- **问题**: 极性材料过滤后召回率下降
- **状态**: ✅ 完成

#### NequIP v7 (code_v6)
- **增强**:
  - 6层SE(3)等变注意力
  - 扩展球谐 (l=0,1,2,3 → 16维)
  - 可学习自适应径向基
  - 多头交叉注意力
  - Focal Loss + 标签平滑
- **状态**: ✅ 完成

#### NequIP v8 (code_v6) ⭐
- **优化**:
  - KDTree加速图构建
  - 4层SE(3)消息传递
  - 16维Bessel径向基
  - SMOTE比例1:3
  - 混合精度训练
- **5折交叉验证结果**:
  - AUC: 98.90% ± 0.65%
  - Accuracy: 98.29% ± 0.42%
  - Recall: 87.79% ± 4.21%
  - Precision: 92.66% ± 2.65%
  - F1: 90.09% ± 2.55%
- **状态**: ✅ 完成

#### NequIP v9 (code_v6)
- **终极架构**:
  - 10层ResNet风格
  - 双路径: 标量+向量
  - CrossAttention融合
  - SE(3)等变层
  - DropPath正则化
  - Cosine Annealing调度
- **状态**: ❌ 失败 (Loss=nan)

---

### 4.4 集成与级联模型

#### Composite V2 (code_v6)
- **架构**: 4个子模型加权融合
  - GCNN: 权重2
  - NequIP: 权重3
  - MLP: 权重1
  - ResNet: 权重2
- **元学习器**: 全连接网络
- **结果**:
  - AUC: 98.17%
  - Accuracy: 97.13%
  - Recall: 90.22%
- **状态**: ✅ 完成

#### Cascade Classifier (code_v6)
- **架构**: 两阶段级联
  - Stage 1: 高召回模型 (阈值0.15)
  - Stage 2: 高精确模型
- **5折交叉验证结果**:
  - AUC: 99.08%
  - Accuracy: 97.99%
  - Recall: 92.32%
- **状态**: ✅ 完成

#### Ultimate V2 (code_v6)
- **结果**:
  - AUC: 99.40%
  - Accuracy: 98.03%
  - Recall: 92.59%
- **状态**: ✅ 完成

#### Ensemble + Cost-Sensitive (code_v6)
- **架构**: 3子模型 + 代价敏感学习
  - High-Recall NequIP
  - High-Precision NequIP  
  - GAT模型
  - 元学习器融合
- **代价敏感Focal Loss**:
  - FN代价: 50
  - FP代价: 1
  - Focal γ: 2.0
- **训练中结果**:
  - Best Recall: 96.24%
  - AUC: ~95.96%
- **状态**: ⏳ 训练中

---

### 4.5 生成模型

#### GAN v1 (code_v5)
- **架构**: 
  - 生成器: 噪声→32维特征
  - 判别器: 特征→分类
- **训练**: 150 epochs
- **结果**:
  - 判别器准确率: 94.93%
  - 分类准确率: 94.23%
- **状态**: ✅ 完成

#### GAN v2 (code_v5)
- **架构**: 64维特征版本
- **训练**: 150 epochs
- **结果**:
  - D Accuracy: 85.6%
  - 分类准确率: 100%
- **状态**: ✅ 完成

#### GAN v3/v4 (code_v5)
- **增强**: t-SNE可视化
- **状态**: ✅ 完成

#### CVAE (code_v5)
- **架构**:
  - 编码器: 特征→潜在空间(32维)
  - 解码器: 潜在空间→特征
  - 条件嵌入: 2类别
- **配置**:
  - β-VAE: KL权重=0.001
  - 隐藏维度: 512
- **训练**: 500 epochs
- **状态**: ✅ 完成

---

### 4.6 逆向设计模型

#### Inverse Design v1 (code_v5)
- **目标**: 从目标属性反推晶格参数
- **结果**:
  - 晶格参数MAE: a=0.76Å, b=0.95Å, c=1.26Å
  - 空间群命中率(±5): 40.9%
  - 元素Top-1准确率: ~30%
- **状态**: ✅ 完成

#### Inverse Design v2 (code_v6)
- **结果**:
  - 元素Top-1: 62.7%
  - 元素Top-3: 75.2%
  - Lattice MSE: 0.0215
- **状态**: ✅ 完成

#### Inverse Design v3 (code_v6)
- **改进**: sqrt变换预测b/a, c/a
- **结果**:
  - 元素Top-1: 61.9%
  - 元素Top-3: 77.0%
  - Ratio MSE (sqrt): 0.0020
- **状态**: ✅ 完成

---

## 5. 性能对比表

### 5.1 分类模型性能排名

| 排名 | 模型 | AUC | Accuracy | Recall | Precision | F1 | 状态 |
|:----:|------|:---:|:--------:|:------:|:---------:|:--:|:----:|
| 1 | **GCNN v5 Ensemble** | 99.72% | ~97% | **99.0%** | 96.9% | ~97.9% | ✅ |
| 2 | Ultimate V2 | 99.40% | 98.03% | 92.59% | - | - | ✅ |
| 3 | Cascade Classifier | 99.08% | 97.99% | 92.32% | - | - | ✅ |
| 4 | NequIP v1 | 99.61% | 97.84% | 96.60% | 95.90% | 96.25% | ✅ |
| 5 | NequIP v8 | 98.90% | 98.29% | 87.79% | 92.66% | 90.09% | ✅ |
| 6 | GCNN v4 | 98.75% | - | **98.2%** | 56.4% | - | ✅ |
| 7 | Composite V2 | 98.17% | 97.13% | 90.22% | - | - | ✅ |
| 8 | Ensemble+CS | ~96% | - | **96.24%** | - | - | ⏳ |
| 9 | GCNN v2 | 98.45% | 95.6% | - | - | - | ✅ |

### 5.2 距离目标的差距

| 指标 | 目标 | 当前最佳 | 差距 | 最佳模型 |
|------|:----:|:-------:|:----:|----------|
| Accuracy | 99% | 98.29% | -0.71% | NequIP v8 |
| Recall | 99% | 99.0% | ✅ | GCNN v5 |
| Precision | 95% | 96.9% | ✅ | GCNN v5 |
| AUC | - | 99.72% | - | GCNN v5 |

---

## 6. 关键发现与经验

### 6.1 成功经验

1. **GCNN v5集成策略有效**
   - 3模型集成显著提升性能
   - 高正样本权重(20)有助于召回率
   - 适度过采样(3×)有帮助

2. **Focal Loss对不平衡有效**
   - α=0.85, γ=1.5 是好的起点
   - 需要配合阈值调整

3. **NequIP的E(3)等变性有价值**
   - 物理对称性先验有助于泛化
   - v1版本recall达到96.6%

4. **级联分类器思路可行**
   - Stage 1低阈值高召回
   - Stage 2精细分类

### 6.2 失败教训

1. **NequIP v9过于复杂**
   - 10层ResNet导致梯度问题
   - Loss=nan表明训练不稳定
   - 解决: 简化架构, 更好初始化

2. **极性材料过滤后召回率下降**
   - v6仅43%召回
   - 原因: 部分FE材料被过滤

3. **代价敏感学习效果有限**
   - FN_cost=50仍未达到99%召回
   - 可能需要更极端的代价比

### 6.3 技术洞察

1. **类别不平衡是核心挑战**
   - 1:36.3的不平衡需要多策略组合
   - 单一方法效果有限

2. **阈值调整很关键**
   - 默认0.5阈值不适用
   - 最优阈值通常在0.1-0.5

3. **集成总是有帮助**
   - 多模型融合稳定提升
   - 元学习器效果好于简单平均

---

## 7. 推荐方案

### 7.1 当前最佳方案: GCNN v5 Ensemble
```
性能: Recall=99.0%, Precision=96.9%, AUC=99.72%
```
**推荐原因**:
- 召回率已达目标
- 精确度超过目标
- 架构相对简单, 训练稳定

### 7.2 进一步提升Accuracy的建议

1. **混合GCNN v5 + NequIP v1**
   - GCNN v5: 高召回(99%)
   - NequIP v1: 高精确(95.9%)
   - 两阶段或并行融合

2. **增加数据**
   - 寻找更多铁电正样本
   - 扩充Materials Project数据

3. **超参数精调**
   - 在最佳模型基础上微调
   - 更长时间训练

### 7.3 未来方向

1. **自监督预训练**
   - 使用大量无标签晶体结构预训练
   - 下游微调分类任务

2. **主动学习**
   - 选择最不确定样本标注
   - 迭代提升模型

3. **物理约束更强的架构**
   - 更深入的对称性建模
   - 铁电物理机制嵌入

---

## 📁 附录: 完整文件列表

### 代码目录
```
wh-ai/
├── code_v1/                    # 传统ML
│   ├── R_F_Feature.py
│   ├── R_F_HGB_SVM_vot.py
│   └── R_F_T_SVM_vot.py
├── code_v2/                    # 深度学习探索
│   ├── nd_NN.py
│   ├── nnd_NN_smote.py
│   ├── nnd_XGBoost.py
│   └── ...
├── code_v3/                    # XGBoost优化
│   ├── v3_XGB_save.py
│   └── nnd_XGBoost_woHE.py
├── code_v5/                    # GNN时代
│   ├── GCNN.py ~ GCNN_v6.py
│   ├── NequIP_Classifier.py
│   ├── FE_GAN.py ~ FE_GAN_v4.py
│   └── FE_CVAE.py
└── code_v6/                    # 高级优化
    ├── train_nequip_v6~v9.py
    ├── train_composite_v2.py
    ├── train_cascade.py
    └── train_ensemble_cost_sensitive.py
```

### 模型目录
```
model_gcnn_v2/ ~ model_gcnn_v6/
model_nequip/ ~ model_nequip_v2/
model_gan/ ~ model_gan_v4/
model_cvae/
model_v3/
invs_dgn_model/ ~ invs_dgn_model_v2/
model_enhanced/
```

### 报告目录
```
reports/
reports_gcnn_v2/ ~ reports_gcnn_v6/
reports_nequip/ ~ reports_nequip_v2/
reports_gan/ ~ reports_gan_v4/
reports_cvae/
reports_inverse/ ~ reports_inverse_v3/
```

---

**报告生成**: GitHub Copilot (Claude Opus 4.5)  
**最后更新**: 2026-01-01
