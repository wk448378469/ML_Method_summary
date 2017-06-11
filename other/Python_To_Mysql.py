# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 19:02:45 2017

@author: kaifeng
"""

import pymysql
import pandas as pd

# 尝试连接数据库
try:
    conn = pymysql.connect(
            user = 'root',
            password = 'wodemima',
            host = '127.0.0.1',
            database = 'sakila'
            )
except:
    print ('连接失败')

# 获取操作游标
cursor = conn.cursor()

# 执行SQL查询语句
sql = 'SELECT * FROM sakila.payment'
cursor.execute(sql)
# 获取记录列表
rows = cursor.fetchall()

# 查询返回的数据类型无法直接转化为数据框
data = []
for i in rows:
    data.append(list(i))

# 获取每一个特征的名称
desc = cursor.description
columns = []
for i in range(7):
    columns.append(desc[i][0])

# 保存成数据框的形式
all_data = pd.DataFrame( data = data , columns = columns)

# 关闭连接
cursor.close()
conn.close()