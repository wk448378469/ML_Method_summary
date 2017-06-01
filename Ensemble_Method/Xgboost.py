# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:25:59 2017

@author: 凯风
"""

from xgboost.sklearn import XGBClassifier,XGBRegressor
from sklearn.datasets import load_boston,load_iris
from sklearn.model_selection import train_test_split,GridSearchCV

# 准备数据
boston_dataset = load_boston()
iris_dataset = load_iris()
X_boston,Y_boston = boston_dataset.data,boston_dataset.target
X_iris,Y_iris = iris_dataset.data,iris_dataset.target
X_boston_train,X_boston_test,Y_boston_train,Y_boston_test = train_test_split(X_boston,Y_boston,test_size=0.3,random_state=0)
X_iris_train,X_iris_test,Y_iris_train,Y_iris_test = train_test_split(X_iris,Y_iris,test_size=0.3,random_state=0)
'''
    xgboost:
        第一点就是，方法使用的是二阶的泰勒展开式，为了与后面的正则项结合起来
        第二点就是，改进树的定义，和样本标签结合在一起，再改进目标函数
        优点：
            正则，有利于降低过拟合
            并行计算，速度快
            灵活性高，可以自定义目标函数和评价标准
            处理缺失值，在其内置过程中
            剪枝的方法，对比看下前剪枝和后剪枝的相关知识
        所有的数学推到过程，找时间全部推到一遍吧
'''
# 分类
xgb_clf = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
xgb_clf.fit(X_iris_train,Y_iris_train,eval_metric='auc')
xgb_clf.predict(X_iris_test)
xgb_clf.feature_importances_
# 回归
xgb_rg = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)
xgb_rg.fit(X_boston_train,Y_boston_train)
xgb_rg.predict(X_boston_test)
xgb_rg.feature_importances_

# 交叉验证选择模型
param = {'max_depth':list(range(3,10,2)),
         'min_child_weight':list(range(1,8,2)),
         'gamma':[i/10.0 for i in range(0,4)],
         'subsample':[i/10.0 for i in range(6,10)],
         'colsample_bytree':[i/10.0 for i in range(6,10)],
         'reg_alpha':[1e-5, 1e-2, 0.1, 1]}
gscv = GridSearchCV(estimator=XGBClassifier(),param_grid=param,scoring='f1_macro',n_jobs=4,iid=False,cv=3)
gscv.fit(X_iris_train,Y_iris_train)
gscv.grid_scores_
gscv.best_score_
gscv.best_params_

'''
   max_depth                        最大树深
   learning_rate                    学习速度，一般在0.01-0.2
   n_estimators                     评估器数量
   silent                           是否打印信息
   objective                        自定义目标函数
       分类                         binary:logistic, multi:softmax, multi:softprob.
       回归                         reg:linear、rmse、mae、logloss、error、merror、mlogloss、auc
   booster                          每次迭代的评估器类型，有：gbtree、gblinear
   n_jobs                           最大CPU核数
   nthread                          和上面的类似，不过推荐使用n_jobs
   gamma                            阈值
   min_child_weight                 样本最小权重和，默认是1，是终止条件
   max_delta_step                   每棵树的权重改变最大步长
   subsample                        默认1，训练第一棵树的时候，按subsample的比例随机抽样作为输入
   colsample_bytree                 和上面类似，不过是对特征抽样的
   colsample_bylevel                用来控制树的每一级的每一次分裂，对列数的采样的占比
   reg_alpha                        L1正则项的系数，
   reg_lambda                       L2正则项的系数
   scale_pos_weight                 平衡正负样本的权重，在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。
   base_score                       不太懂...
   random_state                     随机器对象
   seed                             随机数的种子，默认是0，建议使用上面的random_state
   missing                          数据集中的缺失值是什么
'''