# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 准备数据
iris_dataset = load_iris()
X,Y = iris_dataset.data,iris_dataset.target
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=.3)

'''
    最近邻分类，就是不构建什么内部结构，只存储X，以计算距离为主
        1、K近邻分类，需要用户指定K
            
        2、半径近邻分类，基于每个训练点的固定半径内的近邻数量实现学习，需要用户指定半径
            适用于数据不均匀采样的情况下，如果维度过高则此模型会不太适用
'''

rlf1 = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
rlf2 = RadiusNeighborsClassifier(radius=1.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', outlier_label=None, metric_params=None)
rlf1.fit(trainX,trainY)
rlf2.fit(trainX,trainY)
rlf1.score(testX,testY)
rlf2.score(testX,testY)

'''
    n_neighbors                 近邻数
    radius                      半径
    weights                     权重形式
    algorithm                   计算最邻近的算法
        ball_tree                   ball树
        kd_tree                     KD树
        brute                       暴力搜索
        auto                        根据X，自己选择
    leaf_size                   选择ball树和KD树时的叶尺寸
    p                           不太懂 
    metric                      计算树距离度量
    outlier_label               是否把离群样本点独立出来
    metric_params               不太懂
    n_jobs                      计算性能相关
'''