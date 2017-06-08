# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""

import jieba
import nltk

data_chinese = '当蜘蛛网无情地查封了我的炉台,当灰烬的余烟叹息着贫困的悲哀,我依然固执地铺平失望的灰烬,用美丽的雪花写下：相信未来'
data_english = 'Good muffins cost $3.88\nin New York. Please buy me two of them. Thanks.'

# 中文分词
# 全模式
list(jieba.cut(data_chinese,cut_all=True,HMM=True))     # 返回的是迭代器
# 精确模式
list(jieba.cut(data_chinese,cut_all=False,HMM=False))
# 搜索引擎模式
list(jieba.cut_for_search(data_chinese,HMM=True))


# 英文分词
# 按空格
nltk.tokenize.word_tokenize(data_english,language='english')
# 按空格和符号分
nltk.tokenize.wordpunct_tokenize(data_english)
# 按句子分
nltk.tokenize.sent_tokenize(data_english,language='english')


'''
    sentence        需要分词的字符串
    cut_all         是否采用全模式
    HMM             是否使用隐马尔科夫模型
'''

'''
    其他：
        jieba支持指定用户自己定义的字典，以包含jieba中词库中没有的词
        jieba.load_userdict(filename)，filename是文档路径
        文档形式：
        词语      词频(可忽略)      词性(可忽略)
        云计算         3                i
        数据集         1                n
        ……
        用空格间隔
'''
