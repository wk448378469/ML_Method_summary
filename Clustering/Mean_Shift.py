# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:46:26 2017

@author: 凯风
"""

from sklearn.cluster import MeanShift,estimate_bandwidth
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# 2个特征，1500个样本，3类
centers = [[1,1],[-1,-1],[1,-1]]
X,Y = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6) 

'''
    Mean Shift:
        时间复杂度，O(T*N*log(N))低纬度时，O(T*N^2)高维度时
        优点：
            自动计算簇的数量，在小规模数据集上运行效果良好，使用在图片领域较多，抗噪声好，参数少，速度快
        缺点：
            限定宽度后，会有些浪费，效果不一定好
        算法思路：
            在空间中选择一个样本，以这个点为圆心，以参数bandwidth为半径画出空间圆
            此圆内的所有点为向量，相加生成新的向量(mean shift vector)
            以此向量的终点为圆心，再画圆
            repeat，直到满足约束条件
            由此，圆大致移动到密度最大的地方
            PS.向量的计算公式，可以引入核函数
'''

# 自动计算带宽、创建实例、训练模型，预测，计算后的质心
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
cluster = MeanShift(bandwidth=bandwidth, seeds=None, bin_seeding=True, min_bin_freq=1, cluster_all=True, n_jobs=1)
cluster.fit(X)
Y_pred = cluster.predict(X)
cluster.cluster_centers_

# 画图
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=Y_pred)

'''
    bandwidth               带宽，用于RBF内核
    seeds                   用于初始化内核
    bin_seeding             是否使用离散形式
    min_bin_freq            加快算法的，不太懂
    cluster_all             是否把离群点也计算在簇中
    n_jobs                  使用cpu数量
'''