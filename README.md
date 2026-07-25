# Data Science London + Scikit-learn

> Kaggle Getting Started 竞赛  
> 40 维合成数据的二分类问题  
> 训练集 1000 条，测试集 9000 条

## 最终结果

| 方法 | CV Accuracy | Public Score | Private Score |
|------|:-----------:|:------------:|:------------:|
| GMM(4) + RandomForest | 0.9960 ± 0.0058 | **0.99217** | **0.99160** |

## 解决方案

### 核心思路

该数据集为**合成数据**，由高斯混合模型（GMM）生成。利用 GMM 反向建模数据生成过程，将原始 40 维特征转换为 4 维聚类归属概率，再训练 Random Forest 分类。

### 完整流程

```
                     ┌─────────────┐
                     │  train.csv  │
                     │  (1000×40)  │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐      ┌──────────────┐
                     │   test.csv  │      │trainLabels.csv│
                     │  (9000×40)  │      │   (1000×1)   │
                     └──────┬──────┘      └──────┬───────┘
                            │                    │
                     ┌──────▼──────┐             │
                     │ 合并 train  │             │
                     │  + test     │             │
                     │ (10000×40)  │             │
                     └──────┬──────┘             │
                            │                    │
                     ┌──────▼──────┐             │
                     │  GMM 聚类   │             │
                     │ (无监督)    │             │
                     │ n=4, full   │             │
                     └──────┬──────┘             │
                            │ predict_proba      │
                     ┌──────▼──────┐             │
                     │  GMM 特征   │             │
                     │  train:     │             │
                     │  1000×4     │             │
                     │  test:      │             │
                     │  9000×4     │             │
                     └──────┬──────┘             │
                            │                    │
                     ┌──────▼──────┐             │
                     │ 5 折 CV 评估 │◄────────────┘
                     │ RF n=100    │
                     │ depth=3     │
                     │ acc=0.9960  │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │ 全量训练 RF  │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │ 预测 9000 条 │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │ submission  │
                     │  Id+Solution│
                     └─────────────┘
```

### 各阶段详解

#### Step 1: 数据加载

- `train.csv`: 1000 条 × 40 维，float64，无缺失值
- `trainLabels.csv`: 1000 条标签（0/1）
- `test.csv`: 9000 条 × 40 维
- 类别分布：label 0 = 490, label 1 = 510（基本平衡）

#### Step 2: GMM 特征提取

Gaussian Mixture Model 是一种无监督聚类算法，假设数据由 K 个高斯分布混合生成。

- 将 train（1000）和 test（9000）合并为 10000 条
- 搜索最优参数：
  - n_components ∈ [1, 6]
  - covariance_type ∈ ['spherical', 'tied', 'diag', 'full']
  - 评估指标：BIC（Bayesian Information Criterion）
- 最优：n=4, covariance_type='full', BIC=854029.77
- 用最优 GMM 的 `predict_proba` 将 40 维原始特征 → 4 维聚类归属概率

**为什么有效？** 数据本身就是从 4 个高斯分布混合生成的。GMM 反向建模了生成过程，提取出的 4 个概率值精确刻画了每条样本的簇归属，天然利于分类。

#### Step 3: 5 折交叉验证

对 GMM 特征评估不同深度的 Random Forest：

| max_depth | CV Accuracy |
|:---------:|:-----------:|
| 3 | 0.9960 ± 0.0058 |
| 5 | 0.9960 ± 0.0058 |
| 10 | 0.9950 ± 0.0055 |

选择 RF(n_estimators=100, max_depth=3) 作为最终模型。

#### Step 4: 全量训练

在 1000 条训练集（GMM 4 维特征）上训练最终 Random Forest 模型。

#### Step 5: 推理 & 提交

预测 9000 条测试数据，输出 `Id, Solution` 格式文件。

### 快速运行

```bash
# 安装依赖
pip install numpy pandas scikit-learn

# 下载数据到 input/ 目录
# （从 Kaggle 下载 train.csv, trainLabels.csv, test.csv）

# 运行完整流程
python3 -u train.py
```

输出文件：`output/submission.csv`

### 文件结构

```
├── train.py          # 完整训练流程（加载 → GMM → CV → 训练 → 预测 → 提交）
├── eda.py            # 探索性数据分析（PCA、特征分布、相关性等可视化）
├── .gitignore
├── README.md
├── input/            # 竞赛原始数据
│   ├── train.csv
│   ├── trainLabels.csv
│   └── test.csv
└── output/           # 提交结果
    └── submission.csv
```

### 尝试过的其他方法

| 方法 | 最佳 Public Score | 说明 |
|------|:----------------:|------|
| CatBoost + XGBoost + LightGBM 原始 40 维 | 0.89157 | 树模型直接训练 |
| PCA(12) + CatBoost + XGBoost + LightGBM | 0.91207 | PCA 降维去噪 |
| PCA(12) + SVM(RBF) + CatBoost + LightGBM | 0.93256 | SVM 更适合小样本 |
| PCA(12) + SVM(RBF) 单模 | 0.92958 | 仅 SVM，训练 0.2s |
| **GMM(4) + RandomForest** | **0.99217** | GMM 特征提取 + RF |

### 依赖

- Python ≥ 3.8
- numpy
- pandas
- scikit-learn
