# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 准备数据
iris_dataset = load_iris()
X,Y = iris_dataset.data,iris_dataset.target
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=.3)

'''
    决策树分类：具体的实现看ML in action的代码即可
        优点：
            1、易于理解，可视化
            2、支持缺失值，不需要太多的数据
            3、预测的计算代价低
            4、能处理标称型数据和数值型数据
            5、可以处理多输出问题
            6、可以使用统计测试验证模型
        缺点：
            1、易过拟合
            2、受数据波动性影响较大
            3、本质上是个NP难问题
'''

clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)
clf.fit(trainX,trainY)
clf.score(testX,testY)
preY = clf.predict(testX)

'''
    criterion                           划分策略标准
    splitter                            分割策略
    max_depth                           树的最大深度
    min_samples_split                   拆分内部节点所需的最小样本数
    min_samples_leaf                    叶节点需要的最小样本数
    min_weight_fraction_leaf            叶节点输出的权重
    max_features                        寻找最佳分割时要考虑的特征数量
    random_state                        随机器
    max_leaf_nodes                      最大叶节点数
    min_impurity_split                  阈值
    class_weight                        类标签权重
    presort                             是否预选分配数据了
'''
