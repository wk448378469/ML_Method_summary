# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)
'''
    神经网络回归：
        有输入层、隐藏层、输出层的概念，MLP，一般就是用感知器来做隐藏层
        优点：
            是非线性模型
            可以用在在线学习中
        缺点：
            受权重的影响大
            需要调节很多超参数
            对特征的缩放很敏感
        复杂度：O(n*m*h^k*o*i)
            n——样本数
            m——特征数
            h——隐藏层数
            k——神经元数量
            o——输出神经元
            i——迭代次数
'''
rg = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)
rg.loss_
'''
    hidden_layer_sizes                      第i个元素表示第i个隐藏层中神经元的数量
    activation                              隐藏层的激活函数
        identity                                f(x) = x
        logistic                                f(x) = 1 / (1 + exp(-x))
        tanh                                    f(x) = tanh(x)
        relu                                    f(x) = max(0, x)
    solver                                  权重优化方法
        lbfgs                                   标准牛顿法
        sgd                                     随机梯度下降
        adam                                    随机梯度优化法，没见过...
    alpha                                   L2正则化项的惩罚系数
    batch_size                              使用随机梯度优化法时，minibatch的样本数
    learning_rate                           权重更新的学习速度
    learning_rate_init                      学习速度的初始化
    power_t                                 优化方法为sgd时，然后不太懂了...
    max_iter                                最大迭代次数
    shuffle                                 每次迭代时是否洗牌...
    random_state                            随机器
    tol                                     优化停止机制设定
    verbose                                 是否打印信息
    warm_start                              暂时没啥用
    momentum                                sgd时使用，咋说呢这个，不太懂
    nesterovs_momentum                      不太懂也，sgd或上一个参数大于0时使用
    early_stopping                          验证分数没有变化时是否停止
    validation_fraction                     是否设置交叉验证
    beta_1                                  adam时使用，不太懂
    beta_2                                  adam时使用，不太懂
    epsilon                                 adam时使用，不太懂
'''




