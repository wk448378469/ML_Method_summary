# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:24:59 2017

@author: 凯风
"""

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,log_loss,roc_auc_score,recall_score,accuracy_score,average_precision_score,precision_score

iris_dataset = load_iris()
X,Y = iris_dataset.data,iris_dataset.target
clf = SVC(probability=True,random_state=0)

'''
    对于交叉验证，需要有一个评分标准。
    对于分类：
        accuracy                        就是预测和实际比较，正确的占比
        average_precision               平均正确率，P就是准确度 = TP/TP+FP
        f1                              F1 = 2 * P * R / (P + R)，其中R = TP/TP+FN
        f1_micro                        宏平均，对每一个类统计指标值，然后在对所有类求算术平均值
        f1_macro                        微平均，是对数据集中的每一个实例不分类别进行统计建立全局混淆矩阵，然后计算相应指标
        f1_weighted                     加权平均，先计算每个类的指标，然后通过与真实值比较计算平均值   
        f1_samples                      这个真不懂了...                    
        neg_log_loss                    对数损失，-log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
        precision etc.                  P就是准确度 = TP/TP+FP
        recall etc                      R就是召回率 = TP/TP+FN
        roc_auc                         没有中文名，就是那个曲线下的面积
'''

cross_val_score(clf,X,Y,scoring='accuracy')
#cross_val_score(clf,X,Y,scoring='average_precision')   只用于二分
# cross_val_score(clf,X,Y,scoring='f1')                 只用于二分
cross_val_score(clf,X,Y,scoring='f1_micro')
cross_val_score(clf,X,Y,scoring='f1_macro')
cross_val_score(clf,X,Y,scoring='f1_weighted')
# cross_val_score(clf,X,Y,scoring='f1_samples')         只用于二分
cross_val_score(clf,X,Y,scoring='neg_log_loss')
#cross_val_score(clf,X,Y,scoring='precision')           只用于二分
#cross_val_score(clf,X,Y,scoring='recall')              只用于二分
#cross_val_score(clf,X,Y,scoring='recall')              只用于二分
#cross_val_score(clf,X,Y,scoring='roc_auc')             只用于二分

