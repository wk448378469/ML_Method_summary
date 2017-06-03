# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 准备数据
boston = load_boston()
X,Y = boston.data,boston.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)

'''
    正态过程回归：
        1、假设一，给定X，对Y进行建模，并且要假设对应的这些Y值服从联合正态分布~
        所以重要的是协方差矩阵怎么办
        2、假设二，如果两个样本比较接近or近似，对应的Y值的相关性也较高，所以协方差矩阵是X的函数
        所以接下来重要的问题是根据相关性计算得到的矩阵如何是否合法了~
        就用到了核方法...
        3、对于接下来的新样本呢？
        貌似还要用到贝叶斯回归....
        
        优点：具有概率输出，核函数的广泛性
        缺点：需要依靠整个训练集来预测，对特征数较多的数据集效果不好
        
        具体流程：
            1、选择合适的均值函数和核函数，以及噪声变量
            2、计算训练样本的核矩阵
            3、计算需要预测点与训练样本的核值向量
            4、得出新的高斯分布
            5、求出m和D
            6、得出预测点的估计值和误差
'''

rg = GaussianProcessRegressor(kernel=None, alpha=5e-1, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
rg.fit(X_train,Y_train)
Y_pre = rg.predict(X_test)
rg.score(X_test,Y_test)

'''
    kernel                      指定GP的协方差函数的内核
    alpha                       在拟合期间将值添加到核心矩阵的对角线
    optimizer                   不太懂
    n_restarts_optimizer        用于查找内核参数的优化器的重新启动次数最大化对数边际可能性...
    normalize_y                 目标变量是否标准化了
    copy_X_train                是否把训练集保存在模型对象中
    random_state                随机器
'''