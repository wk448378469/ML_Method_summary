# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:53:08 2017

@author: 凯风
"""

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

X_train = np.array([[1,-1,2],
             [2,0,0],
             [0,1,-1]])

X_test = np.array([[1,1,1],
                   [2,4,2],
                   [3,4,3]])
'''
    数据的标准化：
        分类两种：
            1、减去均值、除以方差
            2、把数值型特征压缩在固定范围内
        适用模型：涉及到梯度下降、涉及到距离计算的模型
        注意事项：训练集和测试集不能单独做标准化,是现在训练集上做标准化，然后transform到测试集上
'''


# 方法1：减去均值，除以方差
# 函数(用的少)
X_scale1 = preprocessing.scale(X_train , axis=0 , with_mean = True , with_std = True , copy = True)
X_scale1.std(axis=0)
X_scale1.mean(axis=0)

# 类概念（用的多）
SS = StandardScaler(copy=True,with_mean=True,with_std=True)   # 创建对象
SS.fit(X_train)                                              # 训练标准化小模型
SS.transform(X_train)                                       # 把训练集标准化
SS.transform(X_test)                                       # 把测试集标准化
SS.mean_
SS.scale_

'''
    X           ——数据集
    axis        ——行索引还是列索引
    with_mean   ——是否做均值处理
    with_std    ——是否做方差处理
    copy        ——标准化是否被复制 
'''


# 方法2：压缩在固定范围内
mms = MinMaxScaler(feature_range=(0, 1), copy=True)
mms.fit(X_train)
mms.transform(X_train)
mms.transform(X_test)

'''
    feature_range         ——压缩范围
    copy                  ——标准化是否被复制 
'''

