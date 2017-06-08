# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from gensim.models import Word2Vec
import jieba,logging
import nltk
import numpy as np 
import pandas as pd 

'''
    通过深度学习，将生成每个词的向量。
    具体的研究看论文或blog吧
    https://segmentfault.com/a/1190000008173404
    这是一篇关于官网给出的教程的翻译，感觉还不错
'''


# 日志配置
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 读取数据
filename = 'C:/Users/dell/Desktop/金瓶梅.txt'
data = open(filename,encoding='utf-8',errors='ignore').readlines()

# 特征处理
data = data[256:]
data = list(filter(lambda s : len(s) > 3 , data))
data = list(map(lambda s : s.strip() , data))
words = map(lambda s : list(jieba.cut(s)) , data)
all_words = []
for each in words:
    all_words.extend(each)

# 获取停用词清单
stopwords_english = nltk.corpus.stopwords.words('english')
filename = 'D:/mygit/ML_Method_summary/Text_Operation/Basic_Operation/stopword.txt'
stopwords_chinese = [line.rstrip() for line in open(filename)]

# 停词处理
word_stoped_middle = [i for i in all_words if i not in stopwords_english]
word_stoped = [i for i in word_stoped_middle if i not in stopwords_chinese]

n = np.unique(word_stoped,return_counts=True)
s = pd.Series(data = n[1],index = n[0])
s = s[6:]
book_words = s.sort_values(ascending=False)
book_words.values
book_words.index

# 出现的词大于100次，且为长度大于1的
book_words[book_words > 200]

# word2vec
# 这里最好传入的是迭代器，不要传入其他类型的数据结构，会训练速度
model = Word2Vec(word_stoped)
model['西']
model.most_similar(positive=['西'],topn=20)


# 还有很多方法，可以调用

'''
    sentences               句子，要向量化的对象
    size                    每个词的向量维数
    alpha                   1                 
    window                  1
    min_count               忽略一些词，用于修建内部的字典
    max_vocab_size          
    sample                  1
    seed                    1
    workers                 装了Cpython有效，用于加快训练
    min_alpha               1
    sg                      1
    hs                      1
    negative                1
    cbow_mean               1
    iter                    1
    null_word               1
    trim_rule               1
    sorted_vocab            1
    batch_words             1
'''
