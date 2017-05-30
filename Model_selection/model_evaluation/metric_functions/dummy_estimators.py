# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:24:59 2017

@author: 凯风
"""

from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.svm import SVC,SVR

iris_dataset = load_iris()
boston_dataset = load_boston()
X_iris,Y_iris = iris_dataset.data,iris_dataset.target
Y_iris[Y_iris!=1] = -1  # svc是+1和-1的，转换一下
X_boston,Y_boston = boston_dataset.data,boston_dataset.target
X_iris_train,X_iris_test,Y_iris_train,Y_iris_test = train_test_split(X_iris,Y_iris,random_state=0)
X_boston_train,X_boston_test,Y_boston_train,Y_boston_test = train_test_split(X_boston,Y_boston,random_state=0)

clf = SVC(kernel='rbf',C=1).fit(X_iris_train,Y_iris_train)
clf.score(X_iris_test,Y_iris_test)

rf = SVR().fit(X_boston_train,Y_boston_train)
rf.score(X_boston_test,Y_boston_test)

'''
    虚拟评估
        就是简单快速的比较不同的分类器，不能用于真是的预测神马的
'''

clf = DummyClassifier(strategy='most_frequent',random_state=0,constant=None)
clf.fit(X_iris_train,Y_iris_train)
clf.score(X_iris_test,Y_iris_test)

rf = DummyRegressor(strategy='mean',constant=None,quantile=None)
rf.fit(X_boston_train,Y_boston_train)
rf.score(X_boston_test,Y_boston_test)

'''
    strategy                    策略
        分类：
            stratified          于训练集的分布选取？部分进行分配
            most_frequent       总是预测最多的标签的样本
            prior               和上面的类似
            uniform             随机预测
            constant            用户提供的标签进行预测
        回归：
            mean                预测样本的平均数的
            median              中位数
            quantile            百分位数
            constant            提供的常规数
    random_state                是否随机
    constant                    应该是strategy选择constant后，在这个字段进行补充
    quantile                    选择百分位后，给出具体数值
'''
