# -*- coding: utf-8 -*-
"""
Created on Sat May 27 19:08:51 2017

@author: 凯风
"""

import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

digits_dataset = load_digits()
X,Y = digits_dataset.data,digits_dataset.target

clf = RandomForestClassifier(n_estimators=20)

param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

random_search = RandomizedSearchCV(clf,param_distributions=param_dist,n_iter=20)

random_search.fit(X,Y)
random_search.best_params_
random_search.best_score_
