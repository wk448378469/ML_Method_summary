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
    鲁棒回归(重点在于解决数据集中的异常问题)
        1、RANSAC，从完整数据集中随机选择一个由正常点组成的子集，并在该子集上拟合模型。
            适合小数据，比TheilSen快点，适合处理在目标变量上的异常值
            可以用在非线性或线性数据集上，具体算法过程看官网吧
        2、TheilSen，广义中位数评估
            适合小数据，适合处理在X上的中等异常，如果特征数增加到一定程度时并没有好过最小二乘
            能接受最大X的29%被破坏
        3、Huber，大于线性损失的被认为是异常值的样本
            如果样本数>>特征数，则最快
        
        如果啥都不懂就用RANSAC....
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