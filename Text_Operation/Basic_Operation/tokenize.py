# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

import nltk        # 英文
import jieba       # 中文

'''
    不说人话的翻译...
    如何理解呢...
    
    英文名称        中文翻译                      理解
    token           令牌             每一个词的起始字符位置和结束位置 
    tokenize        令牌化          对一个字符串的句子，生成每个词的令牌
    tokenizer       令牌解析器                 背后的方法 
'''

# 中文的令牌化
result = jieba.tokenize(u'浙江温州，浙江温州，最大皮革厂，江南皮革厂倒闭了！老板黄鹤吃喝嫖赌，欠下了3.5个亿，带着他的小姨子跑了',mode='search',HMM=True)
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
'''
    unicode_sentence            字符串，只接受Unicode
    mode                        模式：默认模式、搜索模式
    HMM                         是否使用隐马尔科夫模型
'''

# 英文的令牌化
result = nltk.ReppTokenizer('Good muffins cost $3.88\nin New York. Please buy me two of them. Thanks.')
for sent in result.tokenize_sents(result, keep_token_positions=True):
    print (sent)