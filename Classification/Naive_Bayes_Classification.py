# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 准备数据
iris_dataset = load_iris()
X,Y = iris_dataset.data,iris_dataset.target
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=.3)

'''
    朴素贝叶斯
        特点就是快，可以解决高维度的数据
        不同的分类器主要是对P(xi|y)的分布形式不同而已
        高斯分布，P(xi|y) = 1/(2πσ^2)^0.5 * exp(-(xi-μ)^2/2σ^2)
        多项式分布,P(xi|y) = 不写了就
        伯努利分布
        ……
'''

rlf1 = GaussianNB(priors=None)
rlf2 = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
rlf3 = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
rlf1.fit(trainX,trainY)
rlf1.fit(trainX,trainY)
rlf1.fit(trainX,trainY)
rlf1.score(testX,testY)
rlf1.score(testX,testY)
rlf1.score(testX,testY)

'''
    priors                  类标签的先验概率
    class_prior             同上
    alpha                   拉普拉斯平滑参数
    binarize                是否已经做了二值化
    fit_prior               其实有点蒙蔽对于class_prior和fit_prior
'''