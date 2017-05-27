# -*- coding: utf-8 -*-
"""
Created on Sat May 27 14:22:35 2017

@author: 凯风
"""

from sklearn.datasets import load_iris
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score  # 通过交叉验证获得模型分数

from sklearn.model_selection import ShuffleSplit     # 以下这些是不同的交叉验证的方法
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut  # 比较特殊的一个是分组方法，要记得有一个groups数组来保存分类标识
from sklearn.model_selection import TimeSeriesSplit   # 另外一个是关于时间序列的处理

dataSet = load_iris()
X = dataSet.data
Y = dataSet.target
clf = SVC(kernel='linear',C=1)      # 随便创建个分类器

'''
    交叉验证的分数指标：
        就是选取数据集中的一部分进行训练，另一部分来预测
        重要的是cv，选择几折
        次重要的应该算是挑选数据集中部分的方法，可能遇见的有：分层抽样、分组抽样、随机抽样、有放回抽样等等吧
        针对不同的模型应该要选择不同的抽样方法才对
'''

cv1 = ShuffleSplit(n_splits=3,test_size=0.3,random_state=0)
cv2 = StratifiedKFold(n_splits=3,shuffle=False,random_state=None)   # 创建迭代器

scores1 = cross_val_score(clf , X , Y , cv = 5 , groups = None, 
                         scoring = None, n_jobs = 1, 
                         verbose = 0, fit_params = None, 
                         pre_dispatch='2*n_jobs')

scores2 = cross_val_score(clf , X , Y , cv = cv1 , groups = None, 
                         scoring = None, n_jobs = 1, 
                         verbose = 0, fit_params = None, 
                         pre_dispatch='2*n_jobs')

scores3 = cross_val_score(clf , X , Y , cv = cv2 , groups = None, 
                         scoring = None, n_jobs = 1, 
                         verbose = 0, fit_params = None, 
                         pre_dispatch='2*n_jobs')

scores1.mean()
scores2.mean()
scores3.mean()

'''
    estimator           ——模型，未训练的可以
    X                   ——数据集
    y                   ——目标变量
    groups              ——通过数组把数据分成测试集和训练集
    scoring             ——字符串，可以是不同的分数，比如精确度、F1、AUC等等
    cv                  ——最重要的，确定分几折，默认3折，也可以接受迭代器
    n_jobs              ——执行任务的时候，CPU的占用数
    verbose             ——不太了解..
    fit_params          ——一个字典，可以在这里补充模型的某些参数！good
    pre_dispatch        ——CV本身是并行计算，所以这个参数是控制作业数的应该
'''
