# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""


from sklearn.linear_model import BayesianRidge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)
'''
    贝叶斯岭回归：
        优点：
            适用于手头上的小数据
            利用了正则化项对数据进行限制
        缺点：
            模型的推到可能很耗时，所以不写了就。。。
'''
rg = BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)
rg.coef_
rg.intercept_
'''
    n_iter                  迭代次数
    tol                     精度
    alpha_1                 超参数
    alpha_2                 超参数
    lambda_1                超参数
    lambda_2                超参数
    compute_score           每一步都计算目标函数
    fit_intercept           计算截距
    normalize               归一化否
    copy_X                  啊
    verbose                 恩
'''
