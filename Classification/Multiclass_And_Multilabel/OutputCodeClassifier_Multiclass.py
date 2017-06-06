# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.datasets import load_iris
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split

iris_data = load_iris()
X,Y = iris_data.data,iris_data.target
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.3) 

'''
    纠错输出码
        和O-vs-O、O-vs-Rest不太一样的方法
        主要是在欧几里得空间表示
        具体的文本解释，看《机器学习》周治平的那个本里面有提到
'''

clf = LinearSVR(random_state=0)
ovrc = OutputCodeClassifier(clf,code_size=1.5, random_state=None, n_jobs=1)
ovrc.fit(trainX,trainY)
ovrc.predict(testX)
ovrc.code_book_

'''
    estimator           评估器
    code_size           空间尺寸？ 
    random_state        随机器
    n_jobs              CPU的作业数量
'''
