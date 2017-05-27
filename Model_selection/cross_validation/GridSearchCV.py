# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:14:24 2017

@author: 凯风
"""

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,KFold
import numpy as np

'''
    通过交叉验证来获取模型中的某些参数：
        方法有两种：
            1、GridSearchCV          只要你给我的参数我全部组合
            2、RandomizedSearchCV    你给我的参数，我随机组合,有它的好处，见它的文档
        
        GridSearchCV的一般使用方法：
            1、要先划分测试集和训练集
            2、设置要交叉验证的参数集
            3、在训练集上进行模型的训练和参数的搜索
            4、在测试集上查看预测效果
            
            ps:我下面并没有这么做
'''

# 读取数据
iris_dataSet = load_iris()
X = iris_dataSet.data
Y = iris_dataSet.target

# 设定实验次数
num_trials = 30

# 设置超参数字典
param = {'C':[0.001,0.01,0.1,1,10],'gamma':[0.1,0.01]}

# 创建一个模型实例，用高斯核
svc = SVC(kernel='rbf')

# 创建两个数组来对比不同的方法的分数
non_nested_scores = np.zeros(num_trials)
nested_scores = np.zeros(num_trials)

# 开始实验
for i in range(num_trials):
    # 公平起见用一样的参数
    inner_cv = KFold(n_splits=4,shuffle=True,random_state=i)
    outer_cv = KFold(n_splits=4,shuffle=True,random_state=i)
    
    # 非嵌套方法
    clf = GridSearchCV(estimator=svc,param_grid=param,cv=inner_cv)      
    clf.fit(X,Y)
    non_nested_scores[i] = clf.best_score_

    # 嵌套方法
    nested_score = cross_val_score(clf,X=X,y=Y,cv=outer_cv)
    nested_scores[i] = nested_score.mean()

nested_scores.mean()
non_nested_scores.mean()

clf.best_params_    # 获取最佳组合
clf.grid_scores_    # 每次不同超参数组合的分数

'''
    对比：
        非嵌套：容易产生过于乐观的结果
        嵌套：过度应用数据，主要取决于数据集大小和模型的稳定性
'''

'''
    estimator               ——模型
    param_grid              ——超参数字典
    scoring                 ——评分标准
    fit_params              ——其他要传给模型的参数
    n_jobs                  ——不重要，暂时
    iid                     ——是否独立同分布
    refit                   ——可以根据之前是否训练过模型来确定该参数是True还是False
    cv                      ——迭代器，可以穿入不同的自己想要的方法
    verbose                 ——暂时不重要
    pre_dispatch            ——暂时不重要
    error_score             ——如果遇到错误肿么办
    return_train_score      ——
'''