# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)
'''
    简单的线性回归：
        复杂度：O(np^2)
        利用奇异值分解来计算最小二乘的解
'''
rg = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)
rg.coef_
rg.intercept_
'''
    fit_intercept           是否计算截距
    normalize               是否对X进行归一化
    copy_X                  是否覆盖模型中的X
    n_jobs                  CPU的数量，提示计算速度
'''
