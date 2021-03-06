# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from gensim.models import TfidfModel

'''
    一种用于资讯检索与资讯探勘的常用加权技术, 
    主要思想是：
        如果某个词或短语在一篇文章中出现的频率TF高，并且在其他文章中很少出现，
        则认为此词或者短语具有很好的类别区分能力，适合用来分类。
    TF词频(Term Frequency) = 词在文档中的次数/所有词的总数量和
    IDF反文档频率(Inverse Document Frequency) = log(文档数量/(包含词的文档数量+1))
    tfidf = TF * IDF
'''

# 总文档集合
corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],       # 第一个文档，三个词0,1,2,分别出现一次。下面以此类推
          [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
          [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
          [(0, 1.0), (4, 2.0), (7, 1.0)],
          [(3, 1.0), (5, 1.0), (6, 1.0)],
          [(9, 1.0)],
          [(9, 1.0), (10, 1.0)],
          [(9, 1.0), (10, 1.0), (11, 1.0)],
          [(8, 1.0), (10, 1.0), (11, 1.0)]]

tfidf = TfidfModel(corpus)  # 调用方法
vec = [(4,1),(11,1)]
tfidf[vec]

'''
    corpus                      文集
    id2word                     不懂
    dictionary                  指定具体的构造反频率的文集
    normalize                   是否归一化
'''