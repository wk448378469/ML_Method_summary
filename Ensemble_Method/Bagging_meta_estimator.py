# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:25:59 2017

@author: 凯风
"""

from sklearn.ensemble import BaggingClassifier,BaggingRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.datasets import load_boston,load_iris
from sklearn.model_selection import train_test_split

# 准备数据
boston_dataset = load_boston()
iris_dataset = load_iris()
X_boston,Y_boston = boston_dataset.data,boston_dataset.target
X_iris,Y_iris = iris_dataset.data,iris_dataset.target
X_boston_train,X_boston_test,Y_boston_train,Y_boston_test = train_test_split(X_boston,Y_boston,test_size=0.3,random_state=0)
X_iris_train,X_iris_test,Y_iris_train,Y_iris_test = train_test_split(X_iris,Y_iris,test_size=0.3,random_state=0)

'''
    bagging:
        让该学习算法训练多轮，每轮的训练集由从初始的训练集中随机取出的n个训练样本组成，
        某个初始训练样本在某轮训练集中可以出现多次或根本不出现，训练之后可得到一个预测函数序列。
        
        流程：
            1、从样本集中有放回的抽样M个样本。
            2、用这M个样本训练基分类器C。
            3、重复这个过程X次，得到若干个基分类器。
        
        对于分类：
            投票方法
        对于回归：
            平均或者其他方法
        
        整体模型的方差小于等于基模型的方差（当相关性为1时取等号），
        随着基模型数（m）的增多，整体模型的方差减少，从而防止过拟合的能力增强，
        模型的准确度得到提高.
        当基模型数增加到一定程度时，方差的改变对整体方差的作用很小，
        防止过拟合的能力达到极限，这便是准确度的极限了。
        另外，在bagging中的基模型一定要为强模型，
        否则就会导致整体模型的偏差度低，即准确度低。
'''

# 用于分类
clf_bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
clf_bagging.fit(X_iris_train,Y_iris_train)
Y_iris_pre = clf_bagging.predict(X_iris_test)
# 用于回归
reg_bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
reg_bagging.fit(X_boston_train,Y_boston_train)
Y_boston_pre = reg_bagging.predict(X_boston_test)

'''
    base_estimator          基评估器，如果是None的话就是决策树
    n_estimators            基评估器数量
    max_samples             抽样的样本数，可以是整数和浮点数
    max_features            选取几个特征来训练每个评估器
    bootstrap               布尔值，是否有放回
    bootstrap_features      特征是否被放回
    oob_score               是否使用外包样本来估计误差
    warm_start              不太懂
    n_jobs                  没什么用暂时
    random_state            随机器对象
    verbose                 暂时没什么用
'''
