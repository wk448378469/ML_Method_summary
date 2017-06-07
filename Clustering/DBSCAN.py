# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:46:26 2017

@author: 凯风
"""

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 造数据
centers = [[1, 1], [-1, -1], [1, -1]]
X,Y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)
X = StandardScaler().fit_transform(X)

'''
    DBSCAN:
        基于中心的密度聚类，以空间的某个点为圆心，DBSCAN算法求出半径后，区分出：核心点、边界点、噪声点
        
        算法主要思路：
            1、将所有点标记为核心点、边界点、噪声点
            2、删除噪声点
            3、为距离在半径(eps)之内的所有核心点之间赋予一条边
            4、每组连通的核心形成一个簇
            5、将每个边界点指派到一个与之关联的核心点的簇中
        优点：
            抗噪性好、能处理任意形状和大小的簇
        缺点：
            处理某个簇密度特别大时会比较麻烦，对于高纬度的数据也比较困难
        复杂度：
            O(m^2)最坏，O(m)最好，m是点的数量
'''

# 创建对象，训练模型，聚类后的非离群点的bool型数组用于绘制图形，聚类的数量去除离群点
cluster = DBSCAN(eps=0.3, min_samples=10, metric='euclidean', algorithm='auto', leaf_size=30, p=None, n_jobs=1)
cluster.fit(X)
core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)
core_samples_mask[cluster.core_sample_indices_] = True
labels = cluster.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# 绘制图形
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 黑色的是噪声点
        col = 'k'
    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col , markeredgecolor='k', markersize=14)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col , markeredgecolor='k', markersize=6)
plt.title('Number of clusters: %d' % n_clusters_)
plt.show()

'''
    eps                     被划分到同一簇的最短距离
    min_samples             核心点Eps范围内最小样本数
    metric                  真不是很懂...
    algorithm               计算点距离并找到最近邻的算法
        tdtree                  哈
        balltree                哈
        auto                    哈
        brute                   哈
    leaf_size               选择ball、tdtree时的叶的尺寸
    p                       计算点之间的距离的一个参数，计算距离采用闵可夫斯基度量
    n_jobs                  CPU数量
'''