# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:24:59 2017

@author: 凯风
"""

from sklearn.metrics import matthews_corrcoef,precision_recall_curve,\
                            roc_curve,average_precision_score,\
                            roc_auc_score

y_true = [+1, +1, +1, -1]
y_pred = [+1, -1, +1, +1]

'''
    二分分类：
        目标变量只有两个分类，每个样本只可属于其中某一分类
        pre-recall用在不平衡的问题不好，其他两个还好
            1、corrcoef          马修斯相关系数，针对不平衡问题比较好，公式：(TP*TN-FP*FN)/((TP+FP)(TP+FN)(TN+FP)(TN+FN))^0.5
            2、precision         =TP/(TP+FP)，精度
               recall            =TP/(TP+FN)，召回
               thresholds        阈值
            3、fpr               =FP/(FP+TN)，真阳
               tpr               =TP/(TP+FN)，假阳
               thresholds        阈值
            4、average_precision 平均准确度
            5、auc               面积
            PS。4和5也可以用于多标签问题
'''
# 1
matthews_corrcoef(y_true,y_pred,sample_weight=None)
#2
precision, recall, thresholds = precision_recall_curve(y_true,y_pred,pos_label=None, sample_weight=None, drop_intermediate=True)
precision
recall
thresholds
#3
fpr, tpr, thresholds = roc_curve(y_true,y_pred,pos_label=None, sample_weight=None)
fpr
tpr
thresholds
#4
average_precision_score(y_true,y_pred,average='macro',sample_weight=None)
#5
roc_auc_score(y_true,y_pred,average='macro',sample_weight=None)

'''
    y_true                  真实值
    y_pred                  预测值
    y_score                 目标分数，或者叫置信度？
    probas_pred             决策函数给出的预测值
    pos_label               正类标签
    sample_weight           样本权重
    drop_intermediate       丢掉一些次阈值
    average                 有一些不同的方法返回分数，比如加权神马的
'''

