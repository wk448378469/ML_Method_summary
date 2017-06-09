# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from gensim.models.ldamodel import LdaModel
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

def fit_model(corpus,id2word,num_topics):
    # 训练模型
    lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
    lda.print_topics(num_topics)
    return lda

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
    X = load_data(quantity=10000)
    # 简单的清洗下数据
    X = clean_data(X)
    # 获取训练模型前所需要的数据
    corpus,id2word = model_data(X)
    # 训练模型
    model = fit_model(corpus,id2word,20)
    # 预测数据
    test_data = fetch_20newsgroups().data[-1].lower()
    prediction_data(model,id2word,test_data)


'''
    LDA的一些参数
        corpus                  文集
        num_topics              主题数
        id2word                 词和词频对照表
        distributed             强制分布式计算...有点吊啊        
        chunksize               不知道
        passes                  不知道
        update_every            不知道
        alpha                   影响主题文档稀疏性的超参数
        eta                     同上
        decay                   不知道
        offset                  不知道
        eval_every              不知道
        iterations              迭代次数？
        gamma_threshold         γ的阈值
        minimum_probability     最小可能性的限制，默认应该是0
        random_state            随机器
        ns_conf                 不知道
        minimum_phi_value       不知道
        per_word_topics         不知道
        
    LDA的理论学习：
        http://blog.sina.com.cn/s/blog_9e3e549901018kt7.html
        http://www.cnblogs.com/siegfang/archive/2013/01/30/2882391.html
        http://jonathan-huang.org/research/dirichlet/dirichlet.pdf
        暂时真的看不进去了
'''