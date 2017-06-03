# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)
'''
    lasso回归：
        用L1范数，α||w||
        主要的问题在于优化的方法，因为是绝对值，所以没办法使用比如梯度下降等方法
        lasso可能会用到的优化方法：坐标轴下降法和最小角回归法
        特点：
            　Lasso回归使得一些系数变小，甚至还是一些绝对值较小的系数直接变为0，
              因此特别适用于参数数目缩减与参数的选择，因而用来估计稀疏参数的线性模型。
        复杂度：
            O(mn)，坐标轴下降法
            不知道，最小角回归法
'''
rg = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)
rg.coef_
rg.intercept_
'''
    alpha                       l1范数的惩罚系数
    fit_intercept               是否训练截距
    normalize                   归一化否
    precompute                  是否使用Gram矩阵来加速
    copy_X                      是否覆盖模型中的X
    max_iter                    最大迭代次数
    tol                         神马的精度...
    warm_start                  是否使用上一次调用的解决方案作为初始化
    positive                    设置强制系数为正的嘛？
    random_state                随机器
    selection                   每次迭代的时候，是否更新随机系数
'''
