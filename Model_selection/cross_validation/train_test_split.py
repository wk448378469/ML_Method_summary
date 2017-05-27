# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:55:57 2017

@author: 凯风
"""

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston

dataSet = load_boston()
X,Y = dataSet.data,dataSet.target

# 划分训练集和测试集
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.7,random_state=0)
'''
    X               ——数据集
    Y               ——目标变量
    test_size       ——测试集占比
    random_state    ——是否随机？
'''
