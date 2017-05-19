# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 19:02:45 2017

@author: kaifeng
"""

import mysql.connector
import pandas as pd

try:
    conn = mysql.connector.connect(
            user = 'root',
            password = 'wodemima',
            host = '127.0.0.1',
            database = 'sakila'
            )
except:
    print ('连接失败')

cursor = conn.cursor()
sql = 'SELECT * FROM sakila.payment'
cursor.execute(sql)
rows = cursor.fetchall()
desc = cursor.description

columns = []
for i in range(7):
    columns.append(desc[i][0])
    
all_data = pd.DataFrame(rows,columns = columns)

cursor.close()
conn.close()