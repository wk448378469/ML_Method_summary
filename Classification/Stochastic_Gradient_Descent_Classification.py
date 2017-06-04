# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 准备数据
iris_dataset = load_iris()
X,Y = iris_dataset.data,iris_dataset.target
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=.3)

'''
    应用随机梯度下降的回归：
        优点：
            效率高
            容易工程上实施
        缺点：
            有挺多的超参数
            对数据特征缩放敏感
        复杂度：O(knp)，和回归一样的
        在应用该模型时要记得对数据标准化
        迭代次数最好与样本数相关
        ！！另外支持在线学习！！
'''

rlf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, 
                    l1_ratio=0.15, fit_intercept=True, n_iter=5, 
                    shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, 
                    random_state=None, learning_rate='optimal', 
                    eta0=0.0, power_t=0.5, class_weight=None, 
                    warm_start=False, average=False)
rlf.fit(trainX,trainY)
rlf.score(testX,testY)
preY = rlf.predict(testX)

'''
    loss                    损失函数
        hinge                   线性支持向量机
        log                     罗杰斯蒂回归
        modified_huber          对异常值容忍的分类的那个
        squared_hinge           二次惩罚的hinge
        perceptron              感知器
    penalty                 使用神马惩罚
    alpha                   将正则化项扩大的常熟
    l1_ratio                弹性网里面的l1参数
    fit_intercept           是否训练出截距
    n_iter                  其实有点疑问这个参数
    shuffle                 是否洗牌
    verbose                 暂无用
    epsilon                 没懂
    n_jobs                  CPU的使用情况
    random_state            随机器
    learning_rate           学习速度
    eta0                    初始学习速度
    power_t                 反向缩放学习率
    class_weight            类的权重
    warm_start              上一次计算结果为下一次的初始化
    average                 计算平均SGD权重，并存储到coef_中
'''


