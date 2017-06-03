# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)
'''
    岭回归：
        用L2范数，α||w||^2
        复杂度：O(np^2)
'''
rg = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)
rg.coef_
rg.intercept_
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
