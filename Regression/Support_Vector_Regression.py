# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.svm import SVR
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)

'''
    支持向量回归：
        就是支持向量机分类的回归版本，主要推到基本上是类似的
        需要说明的是还有另外两个问题
        SVR、LinearSVR、NuSVR三个不同的支持向量回归
            LinearSVR，就是SVR的参数设定为linear
            SVR，就是本例用到的
            NuSVR，和SVR在一起的，不过有些不同，理论知识后期补充吧
'''

rg = SVR(kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)
rg.support_vectors_

'''
    kernel              核函数的选定
    degree              多项式核的参数，在其他核时无效
    gamma               rbf、poly、sigmoid核的参数，其他核无效
    coef0               poly、sigmoid核的参数，其他核效果弱
    tol                 公差
    C                   惩罚项系数
    epsilon             这个真没搞懂
    shrinking           是否使用缩减启发式
    cache_size          暂时不重要
    verbose             暂时不重要
    max_iter            求结果过程中迭代次数
'''