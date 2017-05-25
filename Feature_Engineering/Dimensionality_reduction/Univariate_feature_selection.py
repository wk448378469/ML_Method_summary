# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:50:30 2017

@author: 凯风
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.datasets import load_iris
import numpy as np

dataSet = load_iris()
X,Y = dataSet['data'],dataSet['target']
X.shape

'''
    计算单个特征的统计量来选择特征：
        SelectKBest——返回前K个特征
        SelectPercentile——返回前多少半分比的特征
        貌似还可以自定义
'''

skb_classification = SelectKBest(chi2,k=2)
skb_classification.fit_transform(X,Y).shape

skb_regression = SelectKBest(f_regression,k=2)
skb_regression.fit_transform(X,Y).shape

'''
    score_func：选择分数函数，记得不是字符串，是需要导入的方法
        对于分类有：chi2(用的多些), f_classif, mutual_info_classif
        对于回归有：f_regression(用的多些), mutual_info_regression
    k：选择特征数
'''