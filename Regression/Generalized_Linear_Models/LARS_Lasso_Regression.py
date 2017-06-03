# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.linear_model import LassoLars
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)

'''
    利用LARS的lasso回归：
        不知道说什么了，因为完全不了解
'''

rg = LassoLars(alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, eps=2.2204460492503131e-16, copy_X=True, fit_path=True, positive=False)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)
rg.coef_
rg.intercept_

'''
    alpha
    fit_intercept 
    verbose
    normalize
    precompute
    max_iter
    eps
    copy_X
    fit_path
    positive
'''