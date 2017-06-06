# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:46:26 2017

@author: 凯风
"""


from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

digits = load_digits(n_class=10)
X = digits.data
Y = digits.target

'''
    凝聚层次聚类：
        除了凝聚的层次聚类以外，还有一个分裂的，不过貌似sklearn没提供
        凝聚层次聚类以树状图的类似于树的图显示，该图显示簇-子簇之间的联系次序。
        基本思路：
            如果有需要计算邻近度矩阵
            1、合并最接近的两个簇
            2、更新邻近性矩阵，以反映新的簇和原来簇之间的邻近性
            repeat，直到剩下一个簇
        复杂度：时间O(m^2logm),空间O(m^2)
        优点：
            可以根据不同的样本权重处理不同大小的簇
        缺点：
            缺乏全局目标，时间和空间复杂度较高
'''

def nudge_images(X, y):
    # 处理数据处理数据
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),.3 * np.random.normal(size=2),mode='constant',).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y
X, Y = nudge_images(X, Y)

# 创建实例、训练模型
X_red = SpectralEmbedding(n_components=2).fit_transform(X)
cluster = AgglomerativeClustering(linkage='ward', n_clusters=10)
cluster.fit(X_red)
labels = cluster.labels_

#画图
x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
X_red = (X_red - x_min) / (x_max - x_min)
plt.figure(figsize=(6, 4))
for i in range(X_red.shape[0]):
    plt.text(X_red[i, 0], X_red[i, 1], str(Y[i]),color=plt.cm.spectral(labels[i] / 10.),fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.tight_layout()

'''
    n_clusters              簇的数量
    affinity                距离计算公式，比如欧几里得、曼哈顿、余弦等等
    memory                  是否缓存树
    connectivity            连接矩阵
    compute_full_tree       这个不太懂，貌似是一个减少计算的参数
    linkage                 簇质心之间的邻近度计算方法(还有一种单链)
        ward                    先合并两个簇，降低SSE，然后计算两个簇质心的距离平方和
        average                 组平均，两个簇的所有点的平均距离
        complete                全链，最远两个样本点的距离
'''