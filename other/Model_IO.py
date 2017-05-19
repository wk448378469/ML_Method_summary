# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 19:02:45 2017

@author: kaifeng
"""


# 模型的保存
model.save('C:/Users/carne/Desktop/ap.pkl')

# 模型的读取
model = models.ldamodel.LdaModel.load('C:/Users/carne/Desktop/ap.pkl')



