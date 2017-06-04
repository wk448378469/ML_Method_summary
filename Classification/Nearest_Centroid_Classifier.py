# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 准备数据
iris_dataset = load_iris()
X,Y = iris_dataset.data,iris_dataset.target
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=.3)

'''
    最近质心分类：
        和KNN很像，通过每个类的数据计算每个类的质心
        然后用这个质心来表示这个类
        算是比较简单的基分类器，参数不多
'''

rlf = NearestCentroid(metric='euclidean', shrink_threshold=None)
rlf.fit(trainX,trainY)
rlf.score(testX,testY)
preY = rlf.predict(testX)

'''
    metric                      计算距离的方法
    shrink_threshold            是否缩小质心以消除特征的阈值
'''