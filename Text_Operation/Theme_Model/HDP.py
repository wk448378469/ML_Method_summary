# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from gensim.models.hdpmodel import HdpModel
from gensim.models.tfidfmodel import TfidfModel
from gensim import matutils,corpora
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import logging
from sklearn.datasets import fetch_20newsgroups

def load_data(quantity=10000):
    dataset = fetch_20newsgroups()
    X = dataset.data[:quantity]
    return X

def clean_data(data):
    new_data = []
    for each in data:
        # 小写化
        each = each.lower()
        # 删除固定格式的无用的字符串
        del_str_position = each.find('lines:') + 6
        each = each[del_str_position:]
        new_data.append(each)
    return new_data

def model_data(data):
    # 文集向量化
    vec = CountVectorizer(min_df=5,stop_words='english')
    train_X = vec.fit_transform(data)
    corpus = matutils.Sparse2Corpus(train_X)
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # 词典
    dictionary = corpora.Dictionary()
    data_seg = []
    for text in data:
        x = nltk.tokenize.word_tokenize(text,language='english')
        data_seg.append(x)
    dictionary.add_documents(data_seg)
    return corpus_tfidf,dictionary

def fit_model(corpus,id2word,num_topics=20):
    # 训练模型
    hdp = HdpModel(corpus=corpus, id2word=id2word)
    hdp.print_topics(num_topics)
    return hdp

def prediction_data(model,id2word,prediction_data):
    pre_doc = prediction_data
    vec_bow = id2word.doc2bow(pre_doc.lower().split())
    vec_lda = model[vec_bow]
    bestSimilarity = {'theme':-1,'similarity':0}
    for i in range(len(vec_lda)):
        if vec_lda[i][1] > bestSimilarity['similarity']:
            bestSimilarity['similarity'] = vec_lda[i][1]
            bestSimilarity['theme'] = vec_lda[i][0]
    print('最相似的主题是：',bestSimilarity['theme'])
    print('相似度为：',bestSimilarity['similarity'])

if (__name__ == 'main'):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # 读取数据
    X = load_data(quantity=5000)
    # 简单的清洗下数据
    X = clean_data(X)
    # 获取训练模型前所需要的数据
    corpus,id2word = model_data(X)
    # 训练模型
    model = fit_model(corpus,id2word,num_topics=20)
    # 预测数据
    test_data = fetch_20newsgroups().data[-1].lower()
    prediction_data(model,id2word,test_data)
'''
    hdp的参数：
        corpus                  文集
        id2word                 词和词频对照表
        max_chunks              不懂
        max_time                定义一个时间停止训练
        chunksize               不懂
        kappa                   学习率
        tau                     减速参数
        K                       二级截断级别
        T                       一级截断级别
        alpha                   二级集中
        gamma                   一级集中
        eta                     狄式主题
        scale                   不懂
        var_converge            不懂
        outputdir               模型输出路径？
        random_state            随机器
        
    hdp的理论学习：
        http://proceedings.mlr.press/v15/wang11a/wang11a.pdf
'''