# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)
'''
    决策树回归：
        就和分类类似的~
        实现方式参考：https://github.com/wk448378469/learn_Machine_Learning/tree/master/Machine_Learning_in_Action/part8_regTree
        
'''
rg = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, presort=False)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)
rg.feature_importances_
rg.tree_        # 底层的树对象
'''
    criterion                           分割时的方法，有均方误差、平均绝对误差
    splitter                            分割策略，有随机分割和最佳分割
    max_depth                           比较重要的一个参数，树的深度
    min_samples_split                   内部节点的最小样本数
    min_samples_leaf                    叶节点的最小样本数
    min_weight_fraction_leaf            需要在叶节点的所有输入样本的权重总和的最小加权分数
    max_features                        寻找特征进行分割时，要考虑的特征数量
    random_state                        随机器
    max_leaf_nodes                      最大叶节点数？
    min_impurity_split                  分裂时的阈值吧
    presort                             是否预先分配数据
'''