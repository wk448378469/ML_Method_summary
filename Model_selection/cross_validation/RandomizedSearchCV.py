# -*- coding: utf-8 -*-
"""
Created on Sat May 27 19:08:51 2017

@author: 凯风
"""

import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# 读取数据
digits = load_digits()
X, y = digits.data, digits.target

# 创建一个模型
clf = RandomForestClassifier(n_estimators=20)

# 定义一个超参数的字典集和
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# 开始随机搜索
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring=None,fit_params=None,
                                   n_jobs=1,iid=True,refit=True,cv=None,verbose=0,
                                   pre_dispatch='2*n_jobs',random_state=None,
                                   error_score='raise',return_train_score=True)

random_search.fit(X, y)
random_search.best_params_  # 获取最佳参数组合
random_search.best_score_   # 最佳组合的得分

'''
    estimator                   ——模型
    param_distributions         ——参数字典
    n_iter                      ——迭代次数
    scoring                     ——评分标准
    fit_params                  ——要传递给模型的其他参数
    n_jobs                      ——并行作业数
    iid                         ——数据是否独立同分布
    refit                       ——之前是否训练过
    cv                          ——迭代器，默认3折
    verbose                     ——不懂
    pre_dispatch                ——没用暂时
    random_state                ——作用可能是控制采用分布的？
    error_score                 ——遇到错误时如何处理
    return_train_score          ——是否返回训练得分
'''