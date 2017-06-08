# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

from nltk.stem import SnowballStemmer
import nltk

'''
    词干处理，在语义上把一个词的不同形式指向同一个概念
'''

# 方法一，创建实例，调用方法，可以选择语言
# SnowballStemmer.languages      查看支持的语言
stemmer1 = SnowballStemmer('english')
stemmer1.stem(u'imaging')
stemmer1.stem('imagine')
stemmer1.stem('buying')

# 方法二，直接使用，无法选择语言
stemmer2 = nltk.stem.snowball.PortugueseStemmer()
stemmer2.stem(u'imaging')
stemmer2.stem('imagine')
stemmer2.stem('buying')

'''
    language                选择语言
    ignore_stopwords        是否忽略停用词
'''