# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.linear_model import Lars
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)

'''
    最小角度回归：
        大致思路：
            先计算当前残差和所有变量的相关系数，找到相关系数最大的变量，假设为X1，
            在这个变量的方向上移动，移动过程中相关系数逐渐减小，直到减小到与X2和
            残差的相关系数相等时，此时的相关系数就是最小的。
        算法步骤：
            1、将样本中心化，标准化。当前残差向量 r = y  ,系数β初始化为0
            2、寻找与残差向量相关系数最大的样本变量 X1
            3、将系数沿着最小二乘解得方向增大，直到另一个样本变量 X2与残差的相关系数和与当前的一样大
            4、两个样本的系数一起沿着新的的方向移动，直到有新的变量被选入
            5、重复234，直到所有变量被选入， 最终，所有变量都被选中，且残差向量r垂直于所有变量，求得最小二乘解。
        优点：
            用于特征数大于样本数时使用
            时间复杂度比较低
            可以快速改造成lasso
        缺点：
            因为模型是对残差进行迭代设计，所以对噪声敏感
        
'''
rg = Lars(fit_intercept=True, verbose=False, normalize=True, precompute='auto', n_nonzero_coefs=500, eps=2.2204460492503131e-16, copy_X=True, fit_path=True, positive=False)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)
rg.coef_
rg.intercept_
'''
    fit_intercept                       是否训练截距
    verbose                             冗长度
    normalize                           归一化否
    precompute                          是否使用Gram矩阵来加速
    n_nonzero_coefs                     非零系数的目标数
    eps                                 精确度，计算某个值时用到
    copy_X                              是否覆盖模型中的X
    fit_path                            不太理解，暂时应该也用不到
    positive                            设置强制系数为正的嘛？
'''
