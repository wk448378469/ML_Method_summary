# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:09:20 2017

@author: 凯风
"""


# 倒入库
import nltk
import jieba

# 文本实例
data_chinese = '当蜘蛛网无情地查封了我的炉台,当灰烬的余烟叹息着贫困的悲哀,我依然固执地铺平失望的灰烬,用美丽的雪花写下：相信未来'
data_english = 'Good muffins cost $3.88\nin New York. Please buy me two of them. Thanks.'
lower_data_english = data_english.lower()

# 文本分词
chinese_word = list(jieba.cut(data_chinese))
english_word = nltk.tokenize.word_tokenize(lower_data_english,language='english')

# 获取停用词清单
stopwords_english = nltk.corpus.stopwords.words('english')
filename = 'D:/mygit/ML_Method_summary/Text_Operation/Basic_Operation/stopword.txt'
stopwords_chinese = [line.rstrip() for line in open(filename)]

# 停词处理
final_english = [i for i in english_word if i not in stopwords_english]
final_chinese = [i for i in chinese_word if i not in stopwords_chinese]

