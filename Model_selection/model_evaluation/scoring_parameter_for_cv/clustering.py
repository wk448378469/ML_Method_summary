# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:24:59 2017

@author: 凯风
"""

from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score,homogeneity_score,\
                            v_measure_score,completeness_score,\
                            adjusted_mutual_info_score
from sklearn.cluster import KMeans
import numpy as np

'''
    homogeneity_score               均一性，在0-1之间，一个簇只能包含一个类别的样本，即只对一个簇中的样本做考虑，被归到其他不管
    completeness_score              完整性，在0-1之间，指同类样本被归类到相同的簇中，这个簇有没有其他样本不管
    v_measure_score                 均一和完整的加权
    adjusted_rand_score             调整兰德指数，ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    adjusted_mutual_info_score      真搞不懂了
'''

n_sample = 1500
random_state = 170
X,Y = make_blobs(n_samples=n_sample,random_state=random_state)
clu = KMeans(n_clusters=2,random_state=random_state)
Y_pred = clu.fit_predict(X)

cross_val_score(clu,X,Y,scoring='adjusted_rand_score')
# cross_val_score(clu,X,Y,scoring='homogeneity_score')          # 以下四个都不能用
# cross_val_score(clu,X,Y,scoring='v_measure_score')
# cross_val_score(clu,X,Y,scoring='completeness_score')
# cross_val_score(clu,X,Y,scoring='adjusted_mutual_info_score')

adjusted_rand_score(Y,Y_pred)
homogeneity_score(Y,Y_pred)
v_measure_score(Y,Y_pred)
homogeneity_score(Y,Y_pred)
adjusted_mutual_info_score(Y,Y_pred)