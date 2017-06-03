# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)

'''
    K近邻回归：
        优点：易于理解、无序评估过多参数
        缺点：对不平衡问题存在较大问题、计算量大
        模型的复杂度取决于algorithm，不同的构造树方法产生不同的复杂度
'''

rg = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)

'''
    n_neighbors             最重要的参数了算是，近邻数
    weights                 权重形式
        uniform                 每个样本均相等
        distance                按距离付给每个样本权重
        [callable]              用户自定义
    algorithm               用于计算最近邻的算法
    leaf_size               上一个参数中，需要一些叶节点数，传递给其中的比如kt-tree
    p                       和下一个参数的一起用的，计算距离的
    metric                  树的距离计算
    metric_params           上一个参数的一起用的
    n_jobs                  暂不重要
'''