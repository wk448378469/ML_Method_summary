# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:25:59 2017

@author: 凯风
"""

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
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
    随机森林：
        sk的随机森林应该不是投票机制，看官网说是平均的方法
        另外就是随机森林可以给出哪个特征最重要的特质
        优点，可以算作是对数据集要求不高吧应该，毕竟对缺失数据和非平衡数据比较稳健
'''
# 训练模型
clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
clf.fit(X_iris_train,Y_iris_train)
clf.predict(X_iris_test)

reg = RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
reg.fit(X_boston_train,Y_boston_train)
reg.predict(X_boston_test)
'''
    n_estimators                    基评估器数量
    criterion                       划分标准
        分类:                       可以有基尼系数\信息增益熵
        回归:                       可以有均方误差\绝对误差
    max_depth                       最大深度
    min_samples_split               分裂所需的最小样本数
    min_samples_leaf                叶节点最小样本数
    min_weight_fraction_leaf        叶节点最小样本权重总值
    max_features                    节点分裂时参与判断的最大特征数
    max_leaf_nodes                  最大叶节点数
    min_impurity_split              阈值
    bootstrap                       是否bootstrap对样本抽样
    oob_score                       是否使用外包样本来估计误差
    n_jobs                          暂不重要
    random_state                    随机器对象	
    verbose                         不重要暂时
    warm_start                      是否热启动，如果是则下一次训练是以追加树的形式进行
    class_weight                    类别的权值
'''