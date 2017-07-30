# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 18:59:16 2017

@author: 凯风
"""

# 最终需要得到的是Y_pred = rfc_pred*weight0 + xgb_pred*weight1 + gb_pred*weight2
# 中的weight0 —— weight2

import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.datasets import load_iris

data = load_iris()
X,Y = data.data,data.target
# 这里的测试集是为了用来寻找最佳的不同模型的权重
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1314)

clfs = []   # 保存模型的

# 第一个用随机森林模型
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
print ('RFC logloss {score}'.format(score=log_loss(Y_test,rfc.predict_proba(X_test))))
clfs.append(rfc)

# 第二个Xgboost
xgb = XGBClassifier()
xgb.fit(X_train,Y_train)
print ('XGB logloss {score}'.format(score=log_loss(Y_test,xgb.predict_proba(X_test))))
clfs.append(xgb)

# 第三个用SVR
gb = GradientBoostingClassifier()
gb.fit(X_train,Y_train)
print ('GB logloss {score}'.format(score=log_loss(Y_test,gb.predict_proba(X_test))))
clfs.append(gb)

# 三组不同的预测结果
predictions = []
for clf in clfs:
    predictions.append(clf.predict_proba(X_test))

def logLossFunc(weights):
    finalPrediction = 0
    for weight , prediction in zip(weights,predictions):
        finalPrediction += weight * prediction
    return log_loss(Y_test,finalPrediction)

startValues = [0.5]*len(predictions) # 其实更好的应该是尝试随机值

cons = ({'type':'eq','fun':lambda w:1-sum(w)})
bounds = [(0,1)] * len(predictions)

res = minimize(logLossFunc,startValues,method='SLSQP',bounds=bounds,constraints=cons)

print ('{message}'.format(message=res['message']))
print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
# 这个地方因为XGBOOST的效果太好过其他两个了，所以随后的得分之比xgboost的得分少了一点点才

print('Best Weights: {weights}'.format(weights=res['x']))
# 获取每个模型的权重

'''
    minimize:
        scipy 提供的一个最小化一个或多个变量的方法。主要参数如下
            fun                 优化所需要用到的计算方法
            x0                  起始权重
            args                其他需要传递给fun的参数
            method              解算器的类型，有很多，自己看文档吧用到的时候
            jac                 是否提供了fun，好像也可以自己选
            hess                如果使用他提供的fun，就可能需要这个参数
            hessp               同上
            bounds              权重的边界
            constraints         约束条件
            tol                 终止条件
        返回一个字典，里面有一些信息，自己看看就能懂
'''











       