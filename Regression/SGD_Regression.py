# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)

'''
    随机梯度下降回归：
        适用于样本数大于10000的数据集上
        其他的话，比如这个数据集，效果是真的差
        优点：
            效率高
            易于实施
        缺点：
            有一些超级参数，所以需要交叉验证
            对特征敏感
        复杂度：
            训练：O(kmn),kmn——迭代次数，n——每个样本的非零属性的平均数
'''

rg = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, average=False)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)

'''
    loss                                    损失函数
        squared_loss                            普通的最小二乘法
        huber                                   Huber损失强劲方法
        epsilon_insensitive                     线性支持向量方法
        squared_epsilon_insensitive             平均线性支持向量方法
    penalty                                 惩罚项，正则化项
    alpha                                   正则化项系数
    l1_ratio                                弹性网中混合参数
    fit_intercept                           数据是否做标准化了
    n_iter                                  迭代次数
    shuffle                                 每次训练后是否洗牌？
    verbose                                 没太懂
    epsilon                                 也是不太懂，不过应该是个重要的
    random_state                            随机器
    learning_rate                           学习率
    eta0                                    初始学习率
    power_t                                 反向缩放学习率的指数
    warm_start                              暂时没用
    average                                 计算平均权重
'''