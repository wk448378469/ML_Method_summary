# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np

# 造点数据
X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
y2 = shuffle(y1, random_state=1)
y3 = shuffle(y1, random_state=2)
Y = np.vstack((y1, y2, y3)).T

# 创建实例
forest = RandomForestClassifier(n_estimators=100, random_state=1)

# 创建实例
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)

# 训练并预测
multi_target_forest.fit(X, Y).predict(X)