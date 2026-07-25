import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

train = pd.read_csv('input/train.csv', header=None)
labels = pd.read_csv('input/trainLabels.csv', header=None).values.ravel()
X = train.values; y = labels
X_test = pd.read_csv('input/test.csv', header=None).values

mean0 = X[y==0].mean(axis=0); mean1 = X[y==1].mean(axis=0)

fig = plt.figure(figsize=(14, 10))

pca = PCA().fit(X)
X_pca = pca.transform(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)

ax1 = plt.subplot(3, 3, 1)
ax1.scatter(X_pca[y==0,0], X_pca[y==0,1], c='#3498db', s=8, alpha=0.5, label='0')
ax1.scatter(X_pca[y==1,0], X_pca[y==1,1], c='#e74c3c', s=8, alpha=0.5, label='1')
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2'); ax1.set_title('PCA'); ax1.legend()

ax2 = plt.subplot(3, 3, 2)
ax2.plot(range(1,41), cumvar, 'b-')
ax2.axhline(0.8, c='gray', ls='--', alpha=0.5)
ax2.axhline(0.95, c='gray', ls='--', alpha=0.5)
ax2.set_xlabel('Components'); ax2.set_ylabel('Cumulative variance'); ax2.set_title('Explained variance')

ax3 = plt.subplot(3, 3, 3)
effect = (mean1 - mean0) / np.sqrt((X[y==0].std(axis=0)**2 + X[y==1].std(axis=0)**2)/2)
colors = ['#e74c3c' if e > 0 else '#3498db' for e in effect]
ax3.bar(range(40), effect, color=colors, width=0.7)
ax3.axhline(0, c='gray', lw=0.5)
ax3.set_xlabel('Feature'); ax3.set_ylabel("Cohen's d"); ax3.set_title('Class separation (effect size)')

feats_to_plot = [0, 4, 12, 14, 18, 6]
for idx, feat in enumerate(feats_to_plot):
    ax = plt.subplot(3, 3, 4 + idx)
    ax.hist(X[y==0,feat], bins=30, alpha=0.6, color='#3498db', label='0', density=True)
    ax.hist(X[y==1,feat], bins=30, alpha=0.6, color='#e74c3c', label='1', density=True)
    ax.set_xlabel(f'Feature {feat}'); ax.legend(fontsize=7)

ax_corr = plt.subplot(3, 3, 7)
corr = np.corrcoef(X.T)
im = ax_corr.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax_corr.set_title('Correlation matrix')
plt.colorbar(im, ax=ax_corr, shrink=0.6)

ax_mean = plt.subplot(3, 3, 8)
ax_mean.scatter(range(40), mean0, c='#3498db', s=15, label='Class 0')
ax_mean.scatter(range(40), mean1, c='#e74c3c', s=15, label='Class 1')
ax_mean.set_xlabel('Feature'); ax_mean.set_ylabel('Mean'); ax_mean.set_title('Feature means by class')
ax_mean.legend()

ax_tt = plt.subplot(3, 3, 9)
train_mean = X.mean(axis=0)
test_mean = X_test.mean(axis=0)
ax_tt.scatter(train_mean, test_mean, s=15, alpha=0.6)
ax_tt.plot([-2,2], [-2,2], 'k--', alpha=0.3)
ax_tt.set_xlabel('Train mean'); ax_tt.set_ylabel('Test mean'); ax_tt.set_title('Train vs Test means')
ax_tt.set_aspect('equal')

plt.tight_layout()
plt.savefig('output/eda.png', bbox_inches='tight', dpi=120)
print('EDA plot saved to output/eda.png')
