# -*- coding: utf-8 -*-
"""
Data Science London + Scikit-learn 二分类竞赛

流程：
  1. 加载数据
  2. GMM 无监督特征提取（在 train+test 合并数据上训练）
  3. 5 折交叉验证评估
  4. 全量训练 & 预测
  5. 输出提交文件

最佳 CV 准确率 ≈ 0.9960
"""

import logging
import time
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

N_FOLDS = 5
RANDOM_SEED = 42

# ============================================================================
# 1. 加载数据
# ============================================================================
log.info('=' * 60)
log.info('Step 1: Loading data')
log.info('=' * 60)

train = pd.read_csv('input/train.csv', header=None).values.astype(float)
y = pd.read_csv('input/trainLabels.csv', header=None).values.ravel().astype(int)
test = pd.read_csv('input/test.csv', header=None).values.astype(float)

log.info('Train features : %s', str(train.shape))
log.info('Test features  : %s', str(test.shape))
log.info('Labels         : %s', dict(zip(*np.unique(y, return_counts=True))))

# ============================================================================
# 2. GMM 无监督特征提取
#    将 train + test 合并（10000 条），搜索最优 GMM
#    GMM 将 40 维原始特征转换为 K 个聚类归属概率（K 维新特征）
# ============================================================================
log.info('')
log.info('=' * 60)
log.info('Step 2: GMM feature extraction')
log.info('=' * 60)

x_all = np.r_[train, test]
log.info('Merged data for GMM fitting: %s', str(x_all.shape))

best_gmm = None
lowest_bic = np.inf
t0 = time.time()

# 搜索最优 GMM：遍历不同簇数（1~6）和协方差类型
for cv_type in ['spherical', 'tied', 'diag', 'full']:
    for n in range(1, 7):
        gmm = GaussianMixture(n_components=n, covariance_type=cv_type, random_state=RANDOM_SEED)
        gmm.fit(x_all)
        bic = gmm.bic(x_all)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

log.info('Best GMM: n_components=%d, covariance_type=%s, BIC=%.2f (%.1fs)',
         best_gmm.n_components, best_gmm.covariance_type, lowest_bic, time.time() - t0)

# 用最优 GMM 将原始 40 维转为 K 维聚类概率
gmm_train = best_gmm.predict_proba(train)
gmm_test = best_gmm.predict_proba(test)
log.info('GMM transformed: train %s, test %s', str(gmm_train.shape), str(gmm_test.shape))

# ============================================================================
# 3. 5 折交叉验证评估
#    在 GMM 特征上评估 RF 性能
# ============================================================================
log.info('')
log.info('=' * 60)
log.info('Step 3: 5-fold cross-validation')
log.info('=' * 60)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# 测试不同树深度的效果
for d in [3, 5, 10]:
    rf = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=RANDOM_SEED, n_jobs=-1)
    scores = cross_val_score(rf, gmm_train, y, cv=cv, scoring='accuracy')
    log.info('  RF depth=%d  accuracy=%.4f ± %.4f', d, scores.mean(), scores.std())

# 选择最优模型
rf_best = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_SEED, n_jobs=-1)
cv_scores = cross_val_score(rf_best, gmm_train, y, cv=cv, scoring='accuracy')
log.info('')
log.info('Best model: RandomForest n=100 max_depth=3')
log.info('CV accuracy: %.4f ± %.4f', cv_scores.mean(), cv_scores.std())

# ============================================================================
# 4. 全量训练 & 预测
# ============================================================================
log.info('')
log.info('=' * 60)
log.info('Step 4: Final training & prediction')
log.info('=' * 60)

t0 = time.time()
rf_best.fit(gmm_train, y)
preds = rf_best.predict(gmm_test)
log.info('Training & prediction completed (%.2fs)', time.time() - t0)

log.info('Prediction distribution: label 0 = %d, label 1 = %d',
         (preds == 0).sum(), (preds == 1).sum())

# ============================================================================
# 5. 输出提交文件
# ============================================================================
log.info('')
log.info('=' * 60)
log.info('Step 5: Saving submission')
log.info('=' * 60)

submission = pd.DataFrame({
    'Id': np.arange(1, len(preds) + 1),
    'Solution': preds,
})
submission.to_csv('output/submission.csv', index=False)
log.info('Submission saved to output/submission.csv')
log.info('')
log.info('Done!')
