# -*- coding: utf-8 -*-
"""
Created on Sat May 27 16:27:43 2017

@author: 凯风
"""

from sklearn.datasets import make_classification       # 自己创建一个数据集
from sklearn.feature_selection import RFECV            # 主要用到的类
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

'''
    通过交叉验证获取最佳特征：
        通过不断构建分类器来测试不同的特征，看指标变换情况，从而决定如何降维
        这里面存在两种办法应该：
            一、从一个特征开始，不断增加特征到下一个模型中
            二、从全部特征开始，不断的减少特征
        问题：
            1、稳定性可能取决于底层的那个模型
            2、这个应该算作是贪心算法吧，所以计算度应该很高的
'''

# 创建一个分类数据集合，1000个样本，25个特征，3核心特征，2个冗余特征，8个分类
X,Y = make_classification(n_samples=1000,n_features=25,n_informative=3,
                          n_redundant=2, n_repeated=0, n_classes=8,
                          n_clusters_per_class=1, random_state=0)

# 创建线性核的svm分类器
svc = SVC(kernel='linear')  

# 调用方法
rfecv = RFECV(estimator=svc,step=1,cv=StratifiedKFold(2),scoring='accuracy')
rfecv.fit(X,Y)

rfecv.n_features_   # 获取核心特征
rfecv.support_      # 特征数量的一维数组，TRUE是核心特征
rfecv.ranking_      # 特征排序
rfecv.grid_scores_  # 特征打分情况

'''
    estimator   ——模型
    step        ——
    cv          ——迭代器
    scoring     ——评分标准
'''





