# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:02:50 2017

@author: 凯风
"""

from sklearn.preprocessing import Imputer
import numpy as np

X_train = np.array([[1,2,2],
                    [np.nan,1,0],
                    [2,0,1]]) 

'''
    缺失数据的补全：
        补充手段：均值、中位数、最频繁的值
        
'''

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0, verbose=0, copy=True)
imp.fit(X_train)
imp.transform(X_train)

'''
    missing_values          ——错误值长成什么鬼样子
    strategy                ——用什么代替，包括：mean、median、most_frequent
    axis                    ——对特征还是对样本
    verbose                 ——真没太懂....
'''