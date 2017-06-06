# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:46:26 2017

@author: 凯风
"""

from sklearn.cluster import birch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 造数据
centers = [[1, 1], [-1, -1], [1, -1]]
X,Y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)
X = StandardScaler().fit_transform(X)

'''
    birch:
        可伸缩的聚类算法 
'''

cluster = birch(threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True)
cluster.fit(X)

'''
    threshold                       1
    branching_factor                1
    n_clusters                      1
    compute_labels                  1
    copy                            1
'''