# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:31:49 2017

@author: 凯风
"""

from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
import numpy as np

dataSet = load_iris()                    
X,Y = dataSet['data'], dataSet['target']
X.shape

'''
    通过树选择特征：
        适用于那些使用到树概念的模型
        特征的选择过程类似于决策树中的计算最佳参数时用到的一些概念，比如信息增益赏、基尼系数
'''

rf = RandomForestRegressor()
rf.fit(X,Y)
rf.feature_importances_                 # 查看变量的重要性和L1base feature selection是类似的作用

sfm = SelectFromModel(rf,prefit=True)
sfm.transform(X).shape

'''
    estimator ——模型
    prefit    ——是否提前训练基模型了
    不需要输入阈值，如果是基于树概念的模型
'''


