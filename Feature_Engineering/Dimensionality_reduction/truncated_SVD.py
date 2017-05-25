# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:41:56 2017

@author: 凯风
"""

from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import load_boston
import numpy as np


dataSet = load_boston()                    
X,Y = dataSet['data'], dataSet['target']


'''
    奇异值分解：
        应该也是用到图片上比较多，如果是降维的话
        如果是SVD应用到推荐系统的话多应该
        可参考之前写过的：https://github.com/wk448378469/learn_Machine_Learning/tree/master/Machine_Learning_in_Action/part13_SVD
'''


tsvd = TruncatedSVD(n_components=4, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
tsvd.fit_transform(X,Y).shape
np.sum(tsvd.explained_variance_ratio_) 


'''
    n_components ——要保留的特征数
    algorithm
    n_iter       ——后面的好像都没太弄懂是神马...
    tol
    random_state
'''