# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 09:18:44 2017

@author: 凯风
"""

import pymysql

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

# 使用cursor()方法获取操作游标 
cursor = conn.cursor()

# select,选取表中的列
sql = 'select id,name from world.city'
cursor.execute(sql)
rows = cursor.fetchall()

# select,选取表中的全部
sql = 'select * from world.city'
cursor.execute(sql)
rows = cursor.fetchall()

# where,对某一行做判定条件后选取
sql = 'select * from world.city where population >= 500000'
cursor.execute(sql)
rows = cursor.fetchall()

# where,对某一行的某一列做判定条件后选取
sql = 'select * from world.city where population >= 500000'
cursor.execute(sql)
rows = cursor.fetchall()

# and-并列条件，两个条件均需要满足，在where中的应用
sql = 'select * from world.city where population >= 500000 and countrycode!= \'AUS\' '
cursor.execute(sql)
rows = cursor.fetchall()

# or-或者条件，两个条件满足一个即可，在where中的应用
sql = 'select * from world.city where population >= 500000 or population <=300000 '
cursor.execute(sql)
rows = cursor.fetchall()

# order by -排序,默认升序，加一个desc后为降序。(字符型也可以)
sql = 'select * from world.city where population >= 500000 order by population desc'
cursor.execute(sql)
rows = cursor.fetchall()

# order by 对多个列进行排序，根据顺序，先对第一列排序，然后再对第二列排序
sql = 'select * from world.city where population >= 500000 order by name,population'
cursor.execute(sql)
rows = cursor.fetchall()

# insert -插入，指定列名(按照每一列的顺序依次填入值即可)，但是要记得数据类型的问题
sql = 'insert into world.city (id,name,countrycode,district,population) values(9999,\'test\',\'AFG\',\'Kabol\',110110)'
cursor.execute(sql)
conn.commit()   # 提交到数据库执行

# insert -插入，不指定列名
sql = 'insert into world.city values(8888,\'test\',\'AFG\',\'Kabol\',110110)'
cursor.execute(sql)
conn.commit()   # 提交到数据库执行

# insert -插入，指定列名，但数据不全(id是这张表的索引，countrycode的可选范围必须在foreignkeys中)
sql = 'insert into world.city (id,name,countrycode) values(7777,\'test\',\'AFG\')'
cursor.execute(sql)
conn.commit()   # 提交到数据库执行

# delete -删除，删除符合条件的行
sql = 'delete from world.city where id=9999'
cursor.execute(sql)
conn.commit()   # 提交到数据库执行

# delete -删除，删除符合条件的行
sql = 'delete from world.city where id=7777'
cursor.execute(sql)
conn.commit()   # 提交到数据库执行

# update -更新，某一行,某一列
sql = 'update world.city set id = 8888 where id = 7777'
cursor.execute(sql)
conn.commit()

# update -更新，某一行,多列
sql = 'update world.city set id = 7777,population=131451 where id = 8888'
cursor.execute(sql)
conn.commit()

# distinct -唯一，每一行，指定的列，返回唯一性组合
sql = 'select distinct countrycode,district from world.city'
cursor.execute(sql)
rows = cursor.fetchall()

# limit -限定，限定固定数量的数据
sql = 'select distinct countrycode,district from world.city limit 100'
cursor.execute(sql)
rows = cursor.fetchall()

# in -满足多个值，比如在where操作时，需要指定固定的值返回
sql = 'select distinct countrycode,district from world.city where countrycode in (\'DZA\',\'AFZ\',\'NLD\') '
cursor.execute(sql)
rows = cursor.fetchall()

# between -数值型在某个范围内,和and 一起搭配使用，因为有and的缘故，最好用括号括起来，和条件判断的and区分开来
sql = 'select * from world.city where population (between 100000 and 300000)'
cursor.execute(sql)
rows = cursor.fetchall()

# like -字符串匹配的，一般和通配符%一起使用
sql = 'select * from world.city where name like \'k%n\' '
cursor.execute(sql)
rows = cursor.fetchall()

# inner join -内连接，总的来说就是拼接不同表中的数据，如果表中有至少一个匹配，则返回行
# 连接两个表
sql = '''
        select a.*,b.name
        from world.city as a 
        inner join 
        world.country as b
        on a.countrycode = b.code         
      '''
cursor.execute(sql)
rows = cursor.fetchall()
# 还包含两个点，一个是as，把缩写为；一个是两个表的连接的key，用on指明

# left join -即使右表中没有匹配，也从左表中返回所有行
# 连接三个表
sql = '''
        select a.* , b.name , c.language
        from 
        world.city as a 
        inner join 
        world.country as b
        on a.countrycode = b.code
        left join
        world.countrylanguage as c
        on a.countrycode = c.countrycode         
       '''
cursor.execute(sql)
rows = cursor.fetchall()
# right join -即使左表中没有匹配，也从右表中返回所有行
# full join -只要其中一个表存在匹配，就返回行

# 上面的另一种写法（稍微加入了一些条件，要记得先筛选，再连接，减少计算），更简洁些看起来
sql = '''
         select a.* , b.name , c.language
         from
         (select * from world.city where population >= 500000) as a
         inner join
         (select * from world.country) as b
         on a.countrycode = b.code
         left join
         (select * from world.countrylanguage) as c
         on a.countrycode = c.countrycode
      '''
cursor.execute(sql)
rows = cursor.fetchall()

# sum函数，求某一列的和，顺便重新命个名
sql = 'select sum(population) as sum_pop from world.city'
cursor.execute(sql)
rows = cursor.fetchall()

# avg函数，求某一列的平均和，顺便重新命个名
sql = 'select avg(population) as avg_pop from world.city'
cursor.execute(sql)
rows = cursor.fetchall()

# count函数，求行数
sql = 'select count(*) from world.city where population >= 400000'
cursor.execute(sql)
rows = cursor.fetchall()

# min函数，求某一列的最小值
sql = 'select min(population) from world.city '
cursor.execute(sql)
rows = cursor.fetchall()

# max函数，求某一列的最大值
sql = 'select max(population) from world.city '
cursor.execute(sql)
rows = cursor.fetchall()

# 上面的函数，结合到group by 一起使用，这个会比较常用
# 虽然mysql支持，groupby和上面的函数不一致，但是最好不要用
# 比如在select后面加上id，name什么的，因为这些数据没有被分组，所以显示出来也没有任何意义可言
sql = '''
        select countrycode,count(*) as num
        from world.city
        group by countrycode
        order by num desc
      '''
cursor.execute(sql)
rows = cursor.fetchall()

# 一个groupby，多个函数是可行的
sql = '''
        select countrycode,
        count(*) as num,
        avg(population) as avg_pop,
        sum(population) as sum_pop,
        std(population) as std_pop
        from world.city
        group by countrycode
        order by num desc
      '''
cursor.execute(sql)
rows = cursor.fetchall()

# 两个groupby，多个函数也是可行的
sql = '''
        select countrycode,district,
        count(*) as num,
        avg(population) as avg_pop,
        sum(population) as sum_pop,
        std(population) as std_pop
        from world.city
        group by countrycode,district
        order by num desc
      '''
cursor.execute(sql)
rows = cursor.fetchall()

# having，group by的where条件判断~~
sql = '''
        select countrycode,district,
        count(*) as num,
        avg(population) as avg_pop,
        sum(population) as sum_pop,
        std(population) as std_pop
        from world.city
        group by countrycode,district
        having avg_pop >= 500000 and std_pop <= 1000
        order by num desc
      '''
cursor.execute(sql)
rows = cursor.fetchall()

# concat -组合，组合多个字符串成为一个字符串
sql = 'select actor_id,concat(first_name,\' \',last_name) as name from sakila.actor'
cursor.execute(sql)
rows = cursor.fetchall()

# length -获取长度，获取字符串的长度
sql = 'select actor_id,length(first_name) from sakila.actor'
cursor.execute(sql)
rows = cursor.fetchall()

# length -获取长度，获取字符串的长度
sql = 'select actor_id,length(first_name) from sakila.actor'
cursor.execute(sql)
rows = cursor.fetchall()

# substring -抽取子字符串,第一个参数是列，第二个是截取的起始位，第三个是终止位
sql = 'select substring(description,3,15) from sakila.film '
cursor.execute(sql)
rows = cursor.fetchall()

# date -截取日期，time -截取时间的函数
sql = 'select *,date(last_update) as d ,time(last_update) as m from sakila.film '
cursor.execute(sql)
rows = cursor.fetchall()

# date_format -转换时间数据的格式
sql = 'select *, date_format(date(last_update),\'%y-%m-%d\') as d from sakila.film '
cursor.execute(sql)
rows = cursor.fetchall()

# datediff -计算两个日期的时间间隔，一般用天比较多
sql = 'select *, datediff(now(),last_update) as date_diff from sakila.film '
# now() 当前时间
cursor.execute(sql)
rows = cursor.fetchall()

# case -控制流，也有if，不过貌似用得少些
sql = '''
        select * ,
        case
            when population >= 1000000 then 'big'
            when population < 1000000 and population >= 500000 then 'medium'
            else 'small'
            end as class
        from world.city
      '''
cursor.execute(sql)
rows = cursor.fetchall()

# 创建表，利用返回的数据
sql = '''
        create table if not exists world.test
        as
        select * ,
        case
            when population >= 1000000 then 'big'
            when population < 1000000 and population >= 500000 then 'medium'
            else 'small'
            end as class
        from world.city
      '''
cursor.execute(sql)

# 创建表，自己创建的一个，需要指定列名、类型，是否能为空等等等~~
sql = '''
         create table if not exists world.test2 
         (
         FIRST_NAME  CHAR(20) NOT NULL,
         LAST_NAME  CHAR(20),
         AGE INT,  
         SEX CHAR(1),
         INCOME FLOAT )
      '''
cursor.execute(sql)

# 删除表
sql = '''
        drop table if exists world.test;
        drop table if exists world.test2
      '''
cursor.execute(sql)

# 关闭链接数据库
conn.close()