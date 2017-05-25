# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:26:49 2017

@author: 凯风
"""

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import LassoCV
import numpy as np

dataSet = load_boston()                    
X,Y = dataSet['data'], dataSet['target']
X.shape


'''
    通过L1选择特征：
        适用于目标函数中使用L1范数的模型
        L1范数就是参数向量或矩阵的各元素的绝对值之和
        但是我记得L0页可以实现稀疏啊...为什么都爱L1呢？L0是NP问题？
        
'''


lasso = LassoCV().fit(X,Y)                  # 找一个使用L1的线性模型,比如lasso等
np.sort(lasso.coef_)                        # 看看系数的重要性
sfm = SelectFromModel(lasso, threshold=0.25, prefit=True)
sfm.transform(X).shape


'''
    estimator ——模型
    treshold  ——阈值
    prefit    ——是否提前训练基模型了
'''
