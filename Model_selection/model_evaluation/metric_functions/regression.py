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
    explained_variance_score            解释方差分数，最分数为1，= 1 - var(y-y^)/var(y)
    mean_absolute_error                 平均绝对误差， = 1/n * ∑ abs(yi - y^i)
    mean_squared_error                  平均平方误差， = 1/n * ∑ (yi - y^i)^2
    median_absolute_error               中位绝对误差， =  median(abs(y1 - y^1)……abs(yn - y^n))
    r2_score                            决定系数，R2， = 1 - (∑(yi-y^i)^2)/(∑(yi - y_平均)^2)
'''

#1
explained_variance_score(y_true, y_pred, sample_weight=None, multioutput='raw_values')
#2
mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='raw_values')
#3
mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='raw_values')
#4
median_absolute_error(y_true, y_pred)
#5
r2_score(y_true, y_pred, sample_weight=None, multioutput=None)

'''
    y_true                      真实值
    y_pred                      预测值
    sample_weight               样本权重
    multioutput                 定义多个输出的分数的汇总
        raw_values              返回一套分数？
        uniform_average         平均加权
        variance_weighted       方差加权
        
'''