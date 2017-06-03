# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""


from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)
'''
    被动侵略性回归：
        优点：
            可以在大数据集上使用，因为是增量学习算法
            另外一个可以实现增量学习的回归是SGDRegressor
        缺点：
            暂时未知
        对于算法的具体过程还不是很清楚，所以暂时作为一个黑箱吧

'''
rg = PassiveAggressiveRegressor(C=1.0, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, loss='epsilon_insensitive', epsilon=0.1, random_state=None, warm_start=False)
rg.fit(X_train,Y_train)
rg.partial_fit(X_train,Y_train)     # 增量学习
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)
rg.coef_
rg.intercept_
'''
    C                           正则化项系数 
    fit_intercept               是否计算截距
    n_iter                      迭代次数
    shuffle                     是否洗牌
    verbose                     哈
    loss                        损失函数
    epsilon                     阈值
    random_state                随机器
    warm_start=False            新的迭代开始后，是否用上一次的最后结果作为初始化
'''
