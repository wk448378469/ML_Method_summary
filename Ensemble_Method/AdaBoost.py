# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:25:59 2017

@author: 凯风
"""

from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
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
    adaboost:
        特点：
            缺点：对离群点比较敏感
            优点：泛化错误率低，参数也不多
        流程：
            1、先通过对N个训练样本的学习得到第一个弱分类器
            2、将分错的样本和其他的新数据一起构成一个新的N个的训练样本，通过对这个样本的学习得到第二个弱分类器
            3、将1和2都分错了的样本加上其他的新样本构成另一个新的N个的训练样本，通过对这个样本的学习得到第三个弱分类器
            4、最终经过提升的强分类器。即某个数据被分为哪一类要由各分类器权值决定。
        分类adaboost实现参考：
            https://github.com/wk448378469/learn_Machine_Learning/tree/master/Machine_Learning_in_Action/part6_boost
'''

clf_adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
clf_adaboost.fit(X_iris_train,Y_iris_train)
clf_adaboost.predict(X_iris_test)

rg_adaboost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)
rg_adaboost.fit(X_boston_train,Y_boston_train)
rg_adaboost.predict(X_boston_test)

'''
    base_estimator          基评估器
    n_estimators            基评估器数量
    learning_rate           学习率，约束每个评估器的贡献，与数量有权衡关系
    algorithm               分类的算法选择，其实是个挺重要的参数...
        SAMME               离散升压算法，用预测错误的样本进行调整
        SAMME.R             true升压算法，用预测类概率进行调整
    loss                    更新权重时的损失函数
        linear              线性
        square              平方
        exponential         指数
    random_state            随机器对象
'''