# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:25:59 2017

@author: 凯风
"""

import xgboost as xgb
# 下载及安装地址教程
# https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=zh
import numpy as np

data = np.random.rand(5,10) 
label = np.random.randint(2, size=5) 
dtrain = xgb.DMatrix( data, label=label)

dtest = dtrain

param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'

evallist  = [(dtest,'eval'), (dtrain,'train')]

num_round = 10
bst = xgb.train( param, dtrain, num_round, evallist )

bst.dump_model('dump.raw.txt')
