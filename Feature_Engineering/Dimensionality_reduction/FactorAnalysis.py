# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:33:55 2017

@author: 凯风
"""

from sklearn.decomposition import FactorAnalysis
from sklearn.datasets import load_boston
import numpy as np

dataSet = load_boston()                    
X,Y = dataSet['data'], dataSet['target']

'''
    因子分析:
        这个我记得在Ng的课程中有讲到的
        x = μ + ΛZ + ε
        特征是由一个隐藏的因子Z生成的
        求解那些参数的时候好像还用到了EM算法
        应用的话，比如商铺形象？找出几个特定的用户关心的诸如：商铺环境神马的特征吧
'''

fa = FactorAnalysis(n_components=4,tol=0.01, copy=True, max_iter=1000, noise_variance_init=None, svd_method='randomized', iterated_power=3, random_state=0)
fa.fit_transform(X,Y).shape

                
'''
    n_componentes       ——Z的维数
    tol
    copy
    max_iter
    noise_variance_init
    svd_method
    iterated_power
    random_state
'''