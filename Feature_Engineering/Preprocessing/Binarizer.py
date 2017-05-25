# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:26:05 2017

@author: 凯风
"""

from sklearn.preprocessing import Binarizer
import numpy as np

X_train = np.array([[1,-1,2],
             [2,0,0],
             [0,1,-1]])

X_test = np.array([[1,1,1],
                   [2,4,2],
                   [3,4,3]])
'''
    数据的二值化：
        适用模型：泊松分布、文本处理
        操作特点：返回对于数值特征的阈值判断
'''

bina = Binarizer(threshold=0.5, copy=True)
bina.fit(X_train)
bina.transform(X_train)

'''
    threshold ——阈值
'''
 