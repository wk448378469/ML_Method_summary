# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.kernel_ridge import KernelRidge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)

'''
    核岭回归：
        在l2正则化的线性模型(岭回归)的基础上，引入了核技术的概念
        在岭回归中，用w* = ∑β*z，也就是β代替w
        代价函数随之替换一下即可
        使用梯度下降求解，β = (λI + K)^-1 * y
        特点：
            对于中型数据集较快，但对于大数据集就很吃力了
            训练时间复杂度O(n^3),挺高的
            预测时间复杂度O(n)
'''

rg = KernelRidge(alpha=1, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)

'''
    alpha               惩罚项系数
    kernel              核函数的选定
    gamma               核函数的中的一个参数项
    degree              多项式核的程度
    coef0               多项式核和sigmoid核中一个参数设定
    kernel_params       核函数的附加参数
'''