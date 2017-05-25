# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:18:29 2017

@author: 凯风
"""

from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
import numpy as np

dataSet = load_boston()                    
X,Y = dataSet['data'], dataSet['target']

'''
    主成分分析：
        使用的话就是...图片？
        主要就是标准化数据，最好先做标准化再掉包
        然后求协方差矩阵
        求特征向量和特征值
        排序删除不要的
        可参考之前写过的：https://github.com/wk448378469/learn_Machine_Learning/tree/master/Machine_Learning_in_Action/part12_PCA
'''


pca = PCA(n_components=4, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
pca.fit_transform(X,Y).shape
np.sum(pca.explained_variance_ratio_)   # 看看能解释多少的方差

'''
    n_components ——要保留的特征数
    copy         ——是否复制
    whiten       ——后面的好像都没太弄懂是神马...
    svd_solver   ——这些应该也可以使用交叉验证来选取最好的
    tol
    iterated
    random_state
'''