# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.gaussian_process import GaussianProcessClassifier 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 准备数据
iris_dataset = load_iris()
X,Y = iris_dataset.data,iris_dataset.target
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=.3)

'''
    高斯过程分类:
        预测采用类概率的形式
        其他的和高斯过程回归也挺相似的
        
'''

rlf = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=1)
rlf.fit(trainX,trainY)
rlf.score(testX,testY)
preY = rlf.predict(testX)
rlf.log_marginal_likelihood_value_

'''
    kernel                          核函数
    optimizer                       传递给核函数的参数集
    n_restarts_optimizer            每次优化时，是否允许从指定的阈值空间中随机抽样开始执行
    max_iter_predict                牛顿法在逼近预测值时的最大迭代次数
    warm_start                      不太懂
    copy_X_train                    永久保存数据集到对象中
    random_state                    随机器
    multi_class                     多类的问题的处理方法，1v1，还是1v剩余还是什么
    n_jobs                          CPU计算的数量
'''