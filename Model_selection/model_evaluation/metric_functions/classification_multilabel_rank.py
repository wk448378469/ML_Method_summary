# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:24:59 2017

@author: 凯风
"""

from sklearn.metrics import accuracy_score,classification_report,\
                            f1_score,fbeta_score,hamming_loss,\
                            jaccard_similarity_score,log_loss,\
                            precision_recall_fscore_support,\
                            precision_score,recall_score,zero_one_loss

y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]

'''
    我想吐。。。
        accuracy_score                          分类精度的得分
        classification_report                   像一个报告似得，里面包含挺多参宿
        f1_score                                就计算F1值，公式忘了不写了
        fbeta_score                             Fβ值，应该是在F1基础上做了些加权神马的
        hamming_loss                            海明距离，看看公式应该挺简单的
        jaccard_similarity_score                杰卡德相似系数，和海明距离类似，分母不一样
        precision_recall_fscore_support         哈
        precision_score                         哈
        recall_score                            哈
        zero_one_loss                           0-1损失，就挺简单得到
        log_loss                                对数损失or逻辑回归损失or交叉熵损失
        
'''
#1
accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
#2
classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2)
#3
f1_score(y_true, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
#4
fbeta_score(y_true, y_pred, beta=0.5, labels=None, pos_label=1, average='weighted', sample_weight=None)
#5
hamming_loss(y_true, y_pred, labels=None, sample_weight=None, classes=None)
#6
jaccard_similarity_score(y_true, y_pred, normalize=True, sample_weight=None)
#7
precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None, pos_label=1, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None)
#8
precision_score(y_true, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
#9
recall_score(y_true, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
#10
zero_one_loss(y_true, y_pred, normalize=True, sample_weight=None)
#11
y = ["spam", "ham", "ham", "spam"]
pred = [[.1, .9], [.9, .1], [.8, .2], [.35, .65]]
log_loss(y, pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)


'''
     参数....
         太多了，自己看官网吧，如果以后用的了
'''