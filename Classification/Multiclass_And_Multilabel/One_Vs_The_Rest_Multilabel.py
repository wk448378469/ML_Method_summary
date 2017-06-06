# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

'''
    One vs The Rest
        多标签的情况下，比如文档的分类问题，可以用到，还算有点用
'''

# 生成数据集合
X,Y = make_multilabel_classification(n_classes=2, n_labels=1, allow_unlabeled=True, random_state=1)
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.3) 

# 创建实例
clf = SVC(kernel='linear')
ovrc = OneVsRestClassifier(clf)

# 训练模型
ovrc.fit(trainX,trainY)

# 预测
preY = ovrc.predict(testX)
ovrc.classes_


