# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:46:26 2017

@author: 凯风
"""

from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 我只想说，跑个模型把计算机搞挂了的模型...
# 而且这个理论推导太多了...要不当黑箱吧...

# 2个特征，1500个样本，3类
centers = [[1, 1], [-1, -1], [1, -1]]
X,Y = make_blobs(n_samples=300,centers=centers,cluster_std=0.5,random_state=0) 

'''
    模型推导:
        http://www.cnblogs.com/vivounicorn/archive/2012/02/10/2343377.html
'''

cluster = SpectralClustering(n_clusters=3, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=1)
cluster.fit(X)
cluster.affinity_matrix_    # 用于聚类的亲和矩阵
Y_pred = cluster.fit_predict(X)

# 画图
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=Y_pred)

