# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:44:05 2017

@author: 凯风
"""

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

X_train = pd.DataFrame([[1,2,2],
                        [1,1,0],
                        [2,0,1]])

'''
    数据编码分类(热独码)：
        适用数据：标称型数据
'''

enc = OneHotEncoder(n_values='auto', categorical_features='all',sparse=True, handle_unknown='error')
enc.fit(X_train)
enc.transform(X_train).toarray()

'''
    n_values                ——每个特征的数量，可以接收数组
    categorical_features    ——哪些特征要做处理
    dtype                   ——数据类型
    sparse                  ——返回矩阵还是数组
    handle_unknown          ——遇到错误如何处理
'''