# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split

iris_data = load_iris()
X,Y = iris_data.data,iris_data.target
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.3) 

'''
    One vs The Rest
        一对多的策略，如果用在多类里，其实没什么卵用，因为sk里面的现有分类器多数都实现了对多类的处理
'''

clf = LinearSVR(random_state=0)
ovrc = OneVsRestClassifier(clf, n_jobs=1)
ovrc.fit(trainX,trainY)
ovrc.predict(testX)

'''
    estimator           评估器
    n_jobs              CPU的作业数量
'''
