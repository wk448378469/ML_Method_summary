# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
import numpy as np

# 准备数据
x = np.arange(100)
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(100,)) + 50. * np.log(1 + np.arange(100))
'''
    等渗回归：
        只能用于1维的数据
        目标函数：min(∑w(y - y^)^2)
        限制条件：ymin = y1^ <= y2^ <= …… yn^ = y^max
'''
rg = IsotonicRegression(y_min=None, y_max=None, increasing=True, out_of_bounds='nan')
rg.fit(x,y)

'''
    y_min               目标变量最小值
    y_max               目标变量最大值
    increasing          不太理解
    out_of_bounds       训练集中有超出训练域以外时如何处理
'''