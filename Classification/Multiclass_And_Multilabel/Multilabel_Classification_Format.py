# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:20:28 2017

@author: 凯风
"""

from sklearn.preprocessing import MultiLabelBinarizer

y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
mlb = MultiLabelBinarizer()
mlb.fit_transform(y)
mlb.classes_

'''
    目标变量的one-hot-encode
    每个目标变量都有可能有不同的标签
    根据所有y的全部抱歉，生成一个y的0-1矩阵，标记每个y包含哪个标签
'''