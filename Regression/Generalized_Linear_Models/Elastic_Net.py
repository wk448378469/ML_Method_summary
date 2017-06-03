# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""


from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)
'''
    弹性网
        主要就是正则化项是L1+L2
        目标函数 = 模型损失 + αβ||w|| + α(1-β)/2 * ||w||^2
        优点：
            多个特征有关联关系的情况下，弹性网是有用的
            集成了ridge和lasso的一些好的特点
        问题：
            计算的复杂度可能高一些了就
        优化方法呢？
'''
rg = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)
rg.coef_
rg.intercept_
'''
    alpha                       两个惩罚项系数的相乘，如果是0就是最小二乘了
    l1_ratio                    混合参数，在[0,1]之间是弹性网的
    fit_intercept               是否训练截距
    normalize                   归一化否
    precompute                  是否使用Gram矩阵来加速
    max_iter                    最大迭代次数
    copy_X                      是否覆盖模型中的X
    tol                         精度
    warm_start                  是否使用上一次调用的解决方案作为初始化
    positive                    设置强制系数为正的嘛？
    random_state                随机器
    selection                   每次迭代的时候，是否更新随机系数
'''
