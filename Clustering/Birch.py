# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:46:26 2017

@author: 凯风
"""

from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 造数据
centers = [[1, 1], [-1, -1], [1, -1]]
X,Y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)
X = StandardScaler().fit_transform(X)

'''
    birch:
        可伸缩的聚类算法，具体的介绍可以看《数据挖掘》第九章其他问题与算法
        算法思路：
            1、通过创建汇总数据的CF树，将数据装入内存中
            2、T增值，然后重新插入叶节点项，由于T增加，某些簇将合并
            3、进行全局聚类，可以使用不同形式的全局聚类(使用所有簇之间的逐对聚类的聚类)
            4、使用第三步发现的簇质心，重新分布数据点，从而发现新的簇集合
        优点：
            可以有效的处理离群点
        缺点：
            不适用于高纬度的数据
            经验：feature大于20，用minibicthKmeans吧
            
'''

cluster = Birch(threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True)
cluster.fit(X)
cluster.subcluster_centers_
'''
    threshold                       阈值
    branching_factor                分支因子，限制节点中子集群的数量
    n_clusters                      聚类的数量，可以不填写
    compute_labels                  是否计算每个训练的标签
    copy                            copy data
'''