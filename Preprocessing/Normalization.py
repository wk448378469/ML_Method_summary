# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:06:11 2017

@author: 凯风
"""

from sklearn.preprocessing import Normalizer
import numpy as np

X_train = np.array([[1,-1,2],
             [2,0,0],
             [0,1,-1]])

X_test = np.array([[1,1,1],
                   [2,4,2],
                   [3,4,3]])
'''
    数据的归一化：
        适用模型：涉及距离、相似度的模型
        注意事项：对每个样本进行归一化操作
'''

norm = Normalizer(norm='l2',copy=True)
norm.fit(X_train)
norm.transform(X_train)
norm.transform(X_test)      #这一步没什么卵用

'''
    norm —— 范数选定，l1、l2、max
'''
 