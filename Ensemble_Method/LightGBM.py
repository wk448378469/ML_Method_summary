# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:37:58 2017

@author: 凯风
"""

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,GridSearchCV

# 读取数据
boston_dataset = load_boston()
X_boston,Y_boston = boston_dataset.data,boston_dataset.target
# 分割数据集
X_boston_train,X_boston_test,Y_boston_train,Y_boston_test = train_test_split(X_boston,Y_boston,test_size=0.3,random_state=0)

'''
    LightGBM
        和xgboost类似，不过更快一些貌似
        和xgboost一样，即支持sklearn风格的代码，也有自己的数据格式，lgb.Dataset,然后lgb.train
        官网上自己说的优点：
            更快的训练效率
            低内存的使用
            更好的准确率。。。。
            支持并行学习
            可处理大规模数据
        最后一个问题就是是否能使用贝叶斯参数搜索了
'''

# 创建模型实例
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.005,
                        n_estimators=20)

# 训练
gbm.fit(X_boston_train,Y_boston_train,
        eval_set=[(X_boston_test,Y_boston_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

# 预测
y_pred = gbm.predict(X_boston_test,num_iteration=gbm.best_iteration)

# 评估
print (mean_squared_error(Y_boston_test,y_pred)**0.5)

# 特征重要性
print (list(gbm.feature_importances_))

# Sklearn 的贪婪搜索获取模型最佳参数
estimator = lgb.LGBMRegressor(num_leaves=31)
param_grid = {
        'learning_rate':[0.01,0.1,0.5],
        'n_estimators':[10,20,30]
        }
gbm = GridSearchCV(estimator,param_grid)
gbm.fit(X_boston_train,Y_boston_train)
print (gbm.best_params_)
print (gbm.best_score_)




