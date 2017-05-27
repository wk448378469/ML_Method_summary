# -*- coding: utf-8 -*-
"""
Created on Sat May 27 15:34:35 2017

@author: 凯风
"""

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score,mean_absolute_error	 # 一些指标，见别的目录
import numpy as np

# 测试分类
iris_data = load_iris()
X_iris = iris_data.data
Y_iris = iris_data.target
svc = SVC()

# 测试回归
boston_data = load_boston()
X_boston = boston_data.data
Y_boston = boston_data.target
lr = LinearRegression()

'''
    通过交叉验证获取预测：
        其实和获取模型分数的方法基本上差不多，只是方法名变了下
'''

predicted_iris = cross_val_predict(svc,X_iris,Y_iris,cv=5,
                                   groups=None, cv=None, n_jobs=1, 
                                   verbose=0, fit_params=None, 
                                   pre_dispatch='2*n_jobs', 
                                   method='predict')

accuracy_score(Y_iris,predicted_iris)

predicted_boston = cross_val_predict(lr,X_boston,Y_boston,cv=5,
                                     groups=None, cv=None, n_jobs=1, 
                                     verbose=0, fit_params=None, 
                                     pre_dispatch='2*n_jobs', 
                                     method='predict')

mean_absolute_error(Y_boston,predicted_boston)