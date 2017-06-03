# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.linear_model import RANSACRegressor,TheilSenRegressor,HuberRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from numpy import inf

boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)

'''
    鲁棒回归
        1、RANSAC
        2、TheilSen
        3、Huber
'''

rg_1 = RANSACRegressor(base_estimator=None, min_samples=None, residual_threshold=None, is_data_valid=None, is_model_valid=None, max_trials=100, stop_n_inliers=inf, stop_score=inf, stop_probability=0.99, residual_metric=None, loss='absolute_loss', random_state=None)
rg_2 = TheilSenRegressor(fit_intercept=True, copy_X=True, max_subpopulation=10000.0, n_subsamples=None, max_iter=300, tol=0.001, random_state=None, n_jobs=1, verbose=False)
rg_3 = HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05)

rg_1.fit(X_train,Y_train)
rg_2.fit(X_train,Y_train)
rg_3.fit(X_train,Y_train)

rg_1.score(X_test,Y_test)
rg_2.score(X_test,Y_test)
rg_3.score(X_test,Y_test)


'''
    alpha                       l2范数的惩罚系数
    fit_intercept               是否训练截距
    normalize                   归一化否
    copy_X                      是否覆盖模型中的X
    max_iter                    最大迭代次数
    tol                         神马的精度...
    solver                      优化方法，有‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’
    random_state                随机器
'''
