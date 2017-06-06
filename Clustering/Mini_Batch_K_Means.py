# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:46:26 2017

@author: 凯风
"""

from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# 2个特征，1500个样本，3类
X,Y = make_blobs(n_samples=3000, cluster_std=0.7) 

'''
    MiniBatchKMeans：
        和k-Means基本上差不多的
        不同的是，Mini是在数据集中选取一部分样本来计算，而不是all in
'''

# 创建实例、训练模型，预测，计算后的质心
cluster = MiniBatchKMeans(n_clusters=3, init='k-means++', max_iter=100, batch_size=100, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
cluster.fit(X)
Y_pred = cluster.predict(X)
cluster.cluster_centers_

# 画图
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=Y_pred)

'''
    n_clusters                  聚类的数量
    init                        初始化质心方法
    max_iter                    最大迭代次数
    batch_size                  小批量的大小，0-100之间
    verbose                     暂无作用
    compute_labels              不太懂...
    random_state                随机器
    tol                         阈值？精度？
    max_no_improvement          收敛检验时的一个参数
    init_size                   随机采样的样本数量
    n_init                      不太理解
    reassignment_ratio          重新分配质心时的最大计数数量分数
'''