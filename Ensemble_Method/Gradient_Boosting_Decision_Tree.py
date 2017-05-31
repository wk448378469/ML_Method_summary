# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:25:59 2017

@author: 凯风
"""

from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
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
    GBDT:
        首先GBDT的基评估器是CART树，所以就有些不错的性质，但也对线性问题存在缺陷了就
        additive model：
            目标函数 = ∑l(y,y^) + ∑Ω(f)
            其中的y^要通过迭代计算，其中包含两个部分，一是上一次迭代的预测值，二是一个新函数
            带入目标函数中，l可以是不同的函数，比如平方损失神马的
            带入后需要用泰勒公式展开，只用一阶(Xgboost用二阶，是为了和后面的正则项某一部件结合)
        简单介绍总结一下：
            1、算法每次迭代生成一颗新的决策树 
            2、在每次迭代开始之前，计算损失函数在每个训练样本点的一阶导数和二阶导数
            3、通过贪心策略生成新的决策树，通过参数公式计算每个叶节点对应的预测值
            4、把新生成的决策树添加到模型中
            
        貌似还有另外一个版本的使用残差迭代树的版本
'''

# 分类
clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, 
                                 n_estimators=100, subsample=1.0, 
                                 criterion='friedman_mse', min_samples_split=2, 
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                 max_depth=3, min_impurity_split=1e-07, 
                                 init=None, random_state=None, 
                                 max_features=None, verbose=0, 
                                 max_leaf_nodes=None, warm_start=False, 
                                 presort='auto')
clf.fit(X_iris_train,Y_iris_train)
clf.score(X_iris_test,Y_iris_test)
clf.predict(X_iris_test)
clf.feature_importances_
# 回归
reg = GradientBoostingRegressor(loss='ls', learning_rate=0.1, 
                                n_estimators=100, subsample=1.0, 
                                criterion='friedman_mse', min_samples_split=2, 
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                max_depth=3, min_impurity_split=1e-07, 
                                init=None, random_state=None, 
                                max_features=None, alpha=0.9,
                                verbose=0, max_leaf_nodes=None, 
                                warm_start=False, presort='auto')
reg.fit(X_boston_train,Y_boston_train)
reg.score(X_boston_test,Y_boston_test)
reg.predict(X_boston_test)
reg.feature_importances_
'''
    loss                            损失函数
        分类                        有罗杰斯蒂损失函数和adaboost一样的损失函数
        回归                        有最小二乘、最小绝对损失、前两者结合、分位数回归
    learning_rate                   学习速率
    n_estimators                    评估器数量
    subsample                       子采样率
    criterion                       判断节点是否继续分裂采用的计算方法(回归和分类也不一样的，参考bagging)
    min_samples_split               分裂所需的最小样本数
    min_samples_leaf                叶节点最小样本数
    min_weight_fraction_leaf        叶节点最小样本权重总值
    max_depth                       最大深度
    min_impurity_split              阈值
    init                            初始子模型
    random_state                    随机器对象
    max_features                    节点分裂时参与判断的最大特征数
    alpha                           alpha为损失函数中的参数
    verbose                         暂不重要
    max_leaf_nodes                  最大叶节点数
    warm_start                      是否热启动，如果是，则下一次训练是以追加树的形式进行
    presort                         是否预排序,预排序可以加速查找最佳分裂点，对于稀疏数据不管用
'''


