# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.svm import SVR
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 准备数据
iris_dataset = load_iris()
X,Y = iris_dataset.data,iris_dataset.target
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=.3)

'''
    支持向量机分类：
        优点：
            在高维度的数据空间依然有效
            特征数大于样本数时仍然有效
            支持向量可以保存在内存中，所以也高效
            强大的核函数
        缺点：
            n>>m，评估效果下降的厉害
            不能提供概率评估
        linearSVC，采用one-vs-rest的方法来处理多类问题时
        对于类别不平衡问题时，SVM可以给不同的标签样本设定不同的权重，通过参数来设定
        优化方法：二次优化算法。具体svm内容看笔记就好
        复杂度：O(n*m^2) 到 O(n*m^3)之间
'''

rlf = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
rlf.fit(trainX,trainY)
rlf.score(testX,testY)
rlf.support_vectors_
preY = rlf.predict(testX)

'''
    kernel              核函数选择
        linear              线性核，<x1,x2>
        poly                多项式核，(r<x1,x2>+b)^d
        rbf                 高斯核，exp(-r|x-x'|^2)
        sigmoid             S核，和logstic类似的函数，tanh(r<x1,x2>+b)
    degree              多项式核的最高次数
    gamma               rbf、poly、sigmoid核的系数
    coef0               仅在poly和sigmoid核时有用的一个系数
    tol                 停止标准的公差
    C                   惩罚项系数
    epsilon             暂时不太懂
    shrinking           是否使用缩减启发式
    cache_size          内核缓存的大小，MB单位
    verbose             暂不用了解
    max_iter            最大迭代次数
'''