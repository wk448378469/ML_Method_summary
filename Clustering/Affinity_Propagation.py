# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:46:26 2017

@author: 凯风
"""

from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# 2个特征，1500个样本，3类
centers = [[1, 1], [-1, -1], [1, -1]]
X,Y = make_blobs(n_samples=300,centers=centers,cluster_std=0.5,random_state=0) 

'''
    AP聚类:
        样本对间发送信息直到收敛来创建簇，用少量样本示例来描述数据集
        信息：
            吸引信息矩阵，数据对象i适合作为另一对象k的聚类中心的程度，表示的是从i到k的消息，初始为0
            附属信息矩阵，数据对象i选择数据对象k作为其据聚类中心的适合程度，表示从k到i的消息，初始为0
        
        然后对着两个矩阵进行迭代，直到满足条件，具体迭代的方法查看书籍吧
        
        优点：
            不需要确定聚类的数量
        缺点：
            复杂度高，O(N^2 * T),T迭代次数，N样本数
        
        适合中小数据集
'''

# 创建实例、训练模型，预测，计算后的质心
cluster = AffinityPropagation(damping=0.5, max_iter=500, convergence_iter=20, copy=True, preference=-50, affinity='euclidean', verbose=False)
cluster.fit(X)
Y_pred = cluster.predict(X)
cluster.cluster_centers_
cluster.cluster_centers_indices_    # 簇中心的指示
# 画图
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=Y_pred)

'''
    damping                     阻尼系数，在[0.5,1]之间
    max_iter                    最大迭代数
    convergence_iter            停止收敛的估计簇数量没有变化的迭代次数
    copy                        复制输入数据
    preference                  很重要，数据点偏好
    affinity                    如何计算亲和度，比如欧氏距离
    verbose                     不懂
'''