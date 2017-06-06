# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:46:26 2017

@author: 凯风
"""

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# 2个特征，1500个样本，3类
X,Y = make_blobs(n_samples=1500,random_state=170) 

'''
    K-means：
        大致思路：
            最基础的聚类方法，需要指定聚类的数量
            初始化质心后，计算每个点到每个初始质心的距离
            根据到最近质心的距离，判定所属簇
            计算同一簇的质心
            repeat，直到质心不再发生变化，或小于阈值？
        适用于：
            通用目的、簇的样本数量较均衡的数据集、簇的数量不太多
        不足：
            对分布规则不均衡、细长的效果不好
            如果维度过高，计算复杂度太高
            另外可能收敛到局部最小，所以要多运行几次
'''

# 创建实例、训练模型，预测，计算后的质心
cluster = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
cluster.fit(X)
Y_pred = cluster.predict(X)
cluster.cluster_centers_

# 画图
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=Y_pred)

'''
    n_clusters                      聚类的数量
    init                            初始化质心方法
        k-mean++                        以“智能”的方法初始聚类质心
        random                          随机
        array                           以数组的形式给定初始质心
    n_init                          不太理解
    max_iter                        最大迭代次数
    tol=0.0001                      阈值？精度？
    precompute_distances            是否预先计算距离
    verbose                         暂无作用
    random_state                    随机器
    copy_x                          模型是否保留x
    n_jobs                          计算使用的计算机CPU
    algorithm                       计算距离所使用的算法？
'''