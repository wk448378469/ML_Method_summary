# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:24:59 2017

@author: 凯风
"""

from sklearn.metrics import cohen_kappa_score,confusion_matrix,hinge_loss
import numpy as np
from sklearn.svm import LinearSVC

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [2, 0, 2, 2, 0, 2]

'''
    多类分类：
        目标变量有多个分类，且每个样本只能归属到一个类中
        cohen_kappa_score       
            kappa系数，用来衡量两种标注结果的吻合程度，标注指的是
            把N个样本标注为C个互斥类别，公式就算了，-1到1,1最好
        confusion_matrix        
            混淆矩阵，每一列是样本的预测分类，每一行是样本的真实分
            类，反之也可以的应该
        hinge_loss              
            铰链损失，边缘最大化，Yw是对真实分类的预测值，Yt是对非
            真实分类预测中的最大值，则L = max（1+Yt-Yw,0）,好像起
            源于二分
'''

cohen_kappa_score(y_true, y_pred, labels=None, weights=None)
confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)

X = np.array([[0], [1], [2], [3]])
Y = np.array([0, 1, 2, 3])
labels = np.array([0, 1, 2, 3])
est = LinearSVC()
est.fit(X, Y)
pred_decision = est.decision_function([[-1], [2], [3]])
y_true = [0,2,3]

hinge_loss(y_true, pred_decision, labels, sample_weight=None)

'''
    y1                  真实值
    y2                  预测值
    labels              标签列表
    weights             加权
    y_true              真实值
    y_pred              预测值
    sample_weight       样本权重
    pred_decision       决策函数给出的预测值
'''


