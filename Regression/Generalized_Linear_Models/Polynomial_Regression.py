# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 17:22:56 2017

@author: 凯风
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

model = Pipeline([('poly',PolynomialFeatures(degree=3)),
                  ('linear',LinearRegression(fit_intercept=False))])

X = np.arange(5)
Y = 3 - 2 * X + X ** 2 - X ** 3
model = model.fit(X[:,np.newaxis],Y)

# 查看模型的w
model.named_steps['linear'].coef_
