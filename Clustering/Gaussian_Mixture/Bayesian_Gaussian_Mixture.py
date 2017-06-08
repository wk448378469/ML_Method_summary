# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:52:15 2017

@author: 凯风
"""

from sklearn.mixture import BayesianGaussianMixture
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 造数据
centers = [[1, 1], [-1, -1], [1, -1]]
X,Y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)
X = StandardScaler().fit_transform(X)

'''
    贝叶斯混合高斯
        在最大期望的基础上，做了些变种。比如加入了很多先验的事情，需要对数据集有更好的理解
        优点：
            自动选择的一些超参数，对参数数量的敏感度低，加入了正则化来约束先验
        缺点：
            速度慢些，超参数需要使用交叉验证从而增加的计算量，包含了模型中存在很多隐藏的偏差
'''

cluster = BayesianGaussianMixture(n_components=3, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None, mean_precision_prior=None, mean_prior=None, degrees_of_freedom_prior=None, covariance_prior=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
cluster.fit(X)
cluster.score(X,Y)

'''
    n_components                            混合成分的数量
    covariance_type                         协方差类型
        spherical                               球面
        diagonal                                对角线
        tied                                    所有矩阵共享相同的一般矩阵
        full                                    全协方差
    tol                                     收敛阈值
    reg_covar                               正则化项，在计算协方差矩阵式的约束
    max_iter                                最大迭代次数
    n_init                                  不太懂
    init_params                             初始化权重的计算方法 
    weight_concentration_prior_type         描述重量浓度类型的字符串
    weight_concentration_prior              ！不太理解，不过应该是最重要的！
    mean_precision_prior                    也不太懂
    mean_prior                              先验的均值分布
    degrees_of_freedom_prior                不太懂
    covariance_prior                        先验的协方差分布
    random_state                            随机器
    warm_start                              加速收敛的
    verbose                                 是否打印日志
    verbose_interval                        在下一次打印之前完成的迭代次数
'''