# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:24:59 2017

@author: 凯风
"""

from sklearn.metrics import coverage_error,\
                            label_ranking_average_precision_score,\
                            label_ranking_loss
import numpy as np

'''
    多标签排序：
        coverage_error                          
            涵盖误差，计算预测结果中平均包含多少真实标签，可以用于二分。
            公式： = 1/n * ∑(max(rank(ij)))
                    rank(ij) = abs({k:f^ik >= f^ij})
        label_ranking_average_precision_score   
            标签排序平均精度，据说比涵盖误差更精细些
            公式:  = 1/n * ∑((1/yi) * ∑(L(ij)/rank(ij)))
                    rank(ij) = abs({k:f^ik >= f^ij})
                    L(ij) = {k:yik = 1,f^ik >= f^ij}
        label_ranking_loss                      
            排序误差，更更精细呗
            公式： = 1/n * ∑(1/(abs(yi)*(nlabel - abs(yi))) * abs(L(ij)))
                    L(ij) = {(k,l):f^ik < f^ij , yik = 1 , yil = 0}
'''

y_true = np.array([[1,0,0],[0,0,1]])
y_score = np.array([[0.75,0.5,1],[1,0.2,0.1]])

coverage_error(y_true, y_score, sample_weight=None)
label_ranking_average_precision_score(y_true, y_score)
label_ranking_loss(y_true, y_score, sample_weight=None)

'''
    y_true              真实值
    y_score             预测值
    sample_weight       样本权重
'''