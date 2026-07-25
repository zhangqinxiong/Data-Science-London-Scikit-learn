# Data Science London + Scikit-learn

Kaggle Getting Started 竞赛 — 40 维合成数据的二分类问题。

## 结果

| 方法 | Public Score | Private Score |
|------|:-----------:|:------------:|
| GMM(4) + RandomForest | **0.99217** | **0.99160** |

## 数据处理流程

1. **缺失值处理** — 数值列用中位数填充，类别列用 `'missing'` 填充（该数据集无缺失）
2. **GMM 特征提取** — 在 train+test 合并数据上训练 Gaussian Mixture Model（BIC 选最优：n=4, full），将 40 维原始特征转换为 4 维聚类归属概率
3. **5 折交叉验证** — 在 GMM 特征上评估 Random Forest 性能（**0.9960 ± 0.0058**）
4. **全量训练 & 预测** — 在 1000 条训练数据上训练最终模型，预测 9000 条测试数据

## 文件结构

```
.
├── train.py          # 完整训练流程
├── input/            # 竞赛数据（CSV）
├── output/           # 提交文件
├── eda.py            # 探索性数据分析
├── .gitignore
└── README.md
```

## 运行

```bash
python3 -u train.py
```

## 依赖

- numpy, pandas, scikit-learn

## 为什么 GMM 有效

该数据集是合成数据，由高斯混合模型生成。GMM 反向建模了数据生成过程，提取出的 4 个聚类概率特征天然利于分类，使得任意简单分类器（RF / 逻辑回归 / SVM）都能达到 99%+ 准确率。
