# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:36:06 2017

@author: 凯风
"""

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def creadteC1(dataSet):
    # apriori算法先创建C1，然后根据支持度得到L1，然后再得到C2，然后得到L2…………直到所有项被添加完orC+1所有都低于支持度了
    C1 = []
    for transaction in dataSet:     # 遍历每一个项
        for item in transaction:    # 遍历项中每一个元素
            if not [item] in C1:    
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))  #fromzenset,创建一个不可变无序的独立元素的集合,与set相比没有add，remove等方法

def scanD(D,Ck,minSupport):
    # 参数：数据集、候选集、最小支持度
    # 用于生成L1
    ssCnt = {}  
    for tid in D:             # 遍历数据集
        for can in Ck:          # 遍历候选集，就是C1
            if can.issubset(tid):       # 如果候选集是数据集的子集(这里是元素级的)
                if not can in ssCnt:    # 计算所有候选集的在数据集的出现次数
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:   # 计算每个候选集的支持度
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList , supportData    # 返回候选集中支持度大于最小支持度的，返回每个候选集的支持度
    
def aprioriGen(Lk,k):
    # 根据候选集生成，生成由K个元素的下一个候选集
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):          # 迭代候选集全体元素
        for j in range(i+1,lenLk):
            # 两两组合
            L1 = list(Lk[i])[:k-2]  
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:    # 如果这两个组合的前K-2个元素相同，为啥是K-2呢？？？？？？减少计算次数？
                retList.append(Lk[i] | Lk[j])   # 添加他们的并集
    return retList

def apriori(dataSet,minSupport=0.5):
    # 接受数据集和支持度，返回符合的频繁集，以及对应全部频繁集的支持度
    C1 = creadteC1(dataSet)
    D = list(map(set,dataSet))
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):    # 迭代生成L2\L3……
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData


def generateRules(L,supportData,minConf=0.7):
    # 接受频繁集，包含频繁集的支持度字典，最小可信度
    bigRuleList = []        # 规则列表
    for i in range(1,len(L)):   # 遍历每个频繁集，去掉只有一个的元素的，因为只有一个元素，推荐就没用了
        for freqSet in L[i]:    # 遍历频繁集每个元素
            H1 = [frozenset([item]) for item in freqSet]    # 建立包含单个元素集合的列表
            if i > 1:
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)     # 调用函数
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)        # 调用函数
    return bigRuleList



def calcConf(freqSet,H,supportData,brl,minConf=0.7):
    # 计算可信度
    # 接受单个频繁集，频繁集的每个元素的列表，支持度的字典，存放规则的列表，可信度阈值
    prunedH = []
    for  conseq in H:   # 遍历频繁集的每个元素
        conf = supportData[freqSet]/supportData[freqSet-conseq]     # 计算每个元素的可信度
        if conf >= minConf:
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq,conseq,conf))    # 保存规则及其可信度
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet,H,supportData,brl,minConf = 0.7):
    # 和上面计算可信度的函数是一样的
    # 目的是为了生成候选规则集合的，类似于13——>02这个规则，可拆分成3——>012和1——>023这样
    m = len(H[0])   
    if len(freqSet) > (m+1):    # 频繁集的元素的数目，是否大于单个集合的元素数目？？？？这TMD是数目
        Hmap1 = aprioriGen(H,m+1)   # 获得不同的m+1个元素的集合
        Hmap1 = calcConf(freqSet,Hmap1,supportData,brl,minConf)     # 计算可信度
        if len(Hmap1) > 1:  # 如果新生成的还有两个以上，则继续调用自己
            rulesFromConseq(freqSet,Hmap1,supportData,brl,minConf)
