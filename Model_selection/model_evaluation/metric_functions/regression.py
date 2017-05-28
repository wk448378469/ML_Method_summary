# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:24:59 2017

@author: 凯风
"""

from sklearn.metrics import explained_variance_score,\
                            mean_absolute_error,\
                            mean_squared_error,\
                            median_absolute_error,\
                            r2_score
                            
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

'''
    explained_variance_score            -
    mean_absolute_error                 -
    mean_squared_error                  -
    median_absolute_error               -
    r2_score                            -
'''

#1
explained_variance_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
#2
mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
#3
mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
#4
median_absolute_error(y_true, y_pred)
#5
r2_score(y_true, y_pred, sample_weight=None, multioutput=None)

'''
    y_true              真实值
    y_pred              预测值
    sample_weight       样本权重
    multioutput         定义多个输出的分数的汇总
        raw_values：
        uniform_average：
        variance_weighted：
        
'''