# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:15:40 2017

@author: 凯风
"""

from sklearn.feature_selection import VarianceThreshold
import numpy as np

X_train = np.array([[0, 0, 1], 
                    [0, 1, 0], 
                    [1, 0, 0], 
                    [0, 1, 1], 
                    [0, 1, 0], 
                    [0, 1, 1]])

'''
    根据方差的大小删除特征：
        通过设定阈值（方差值）来进行特征选择
        对线性模型使用       
'''

sel = VarianceThreshold(threshold=0.16)
sel.fit(X_train)
sel.transform(X_train)

'''
    threshold ——方差阈值
'''

