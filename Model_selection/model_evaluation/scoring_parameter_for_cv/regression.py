# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:24:59 2017

@author: 凯风
"""

from sklearn.metrics import mean_absolute_error,\
                            mean_squared_error,\
                            median_absolute_error,\
                            r2_score

'''
    回归：
        neg_mean_absolute_error         ——平均绝对误差回归损失
        neg_mean_squared_error          ——均方误差回归损失
        neg_median_absolute_error       ——中值绝对误差回归损失
        r2                              ——相关系数，最佳分数是1，可以是负的，有点长公式
'''

from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score

boston_dataset = load_boston()
X,Y = boston_dataset.data,boston_dataset.target
reg = Lasso(alpha=0.1)

cross_val_score(reg,X,Y,scoring='neg_mean_absolute_error')
cross_val_score(reg,X,Y,scoring='neg_mean_squared_error')
cross_val_score(reg,X,Y,scoring='neg_median_absolute_error')
cross_val_score(reg,X,Y,scoring='r2')

reg.fit(X,Y)
Y_pred = reg.predict(X)

mean_absolute_error(Y,Y_pred)
mean_squared_error(Y,Y_pred)
median_absolute_error(Y,Y_pred)
r2_score(Y,Y_pred)