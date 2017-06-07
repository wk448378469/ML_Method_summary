# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:52:15 2017

@author: 凯风
"""

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 造数据
centers = [[1, 1], [-1, -1], [1, -1]]
X,Y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)
X = StandardScaler().fit_transform(X)

'''
    高斯混合：
        假定数据由一个统计过程生成，并通过找到最佳拟合数据的统计模型来描述数据，其中统计模型用分布和该分布的一组参数描述。
        EM算法思路：
            1、选择模型参数的初始集
            2、E-step：对于每个对象，计算每个对象属于每个分布的概率，即计算Prob(分布j|xi，Θ)
            3、M-step：最大化步，给定期望步得到的概率，找到最大化该期望似然的新的参数估计
            4、直到参数不再改变
        优点：
            速度快
        缺点：
            对于某个分布中的点不足时，无法计算出协方差矩阵，除非人为的限制
            模型需要计算的组件非常多，并且需要保存的也多
'''

cluster = GaussianMixture(n_components=3, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
cluster.fit(X)
cluster.covariances_
cluster.score(X,Y)

'''
    n_components                    混合成分的数量
    covariance_type                 协方差类型
        spherical                       球面
        diagonal                        对角线
        tied                            所有矩阵共享相同的一般矩阵
        full                            全协方差
    tol                             收敛阈值
    reg_covar                       正则化项，在计算协方差矩阵式的约束
    max_iter                        最大迭代次数
    n_init                          不太懂
    init_params                     初始化权重的计算方法 
        kmeans                          就kmeans
        random                          随机
    weights_init                    用户自己提供初始权重
    means_init                      用户自己提供的初始化权重的计算方法
    precisions_init                 用户提供初始化协方差矩阵的逆
    random_state                    随机器
    warm_start                      加速收敛的
    verbose                         是否打印日志
    verbose_interval                在下一次打印之前完成的迭代次数
'''