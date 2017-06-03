# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:29:05 2017

@author: 凯风
"""

class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue       # 节点名称的变量
        self.count = numOccur       # 计数值
        self.nodeLink = None        # 相似元素项
        self.parent = parentNode    # 父节点
        self.children = {}          # 子节点，可能有很多
    
    def inc(self,numOccur):
        # 给count增加给定值
        self.count += numOccur
    
    def disp(self,ind=1):
        # 把树用文本得到形式所打印
        # ind用来打印N倍的空格符,用此来表示节点的深度
        print ('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

def createTree(dataSet,minSup = 1):
    # 接收的数据集必须经过格式化处理
    headerTabel = {}
    for trans in dataSet:   # 遍历每个样本
        for item in trans:  # 遍历每个样本中每个元素
            # 通过一个空字典来存放每个元素出现的次数来作为头指针
            headerTabel[item] = headerTabel.get(item,0) + dataSet[trans]  
    for k in list(headerTabel):
        # 删除低于支持度的元素项
        if headerTabel[k] < minSup:
            del(headerTabel[k])
            
    freqItemSet = set(headerTabel.keys())   # 保存符合要求的键值
    
    if len(freqItemSet) == 0:
        # 退出，如果没有元素项满足
        return None,None
    for k in headerTabel:
        # 字典中的value值，对应增加一个None（可能是为了满足自定义的数据格式？）
        headerTabel[k] = [headerTabel[k],None]  
        
    retTree = treeNode('Null Set',1,None)   # 初始化第一个节点
    
    for tranSet , count in dataSet.items(): # 迭代每一个样本和次数
        localD = {}
        for item in tranSet:            # 每一个样本中的元素
            if item in freqItemSet:     # 对于之前过滤剩下的样本元素 
                localD[item] = headerTabel[item][0]     # 获得次数
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
            updateTree(orderedItems,retTree,headerTabel,count)  # 使用排序后的次数，元素集对树进行填充
    return retTree,headerTabel

def updateTree(items,inTree,headerTabel,count):
    # 生成树的函数
    # 接收排序后元素集，头指针表，定制的数据类型的数据，次数
    if items[0] in inTree.children:
        # 元素集第一项是否作为子节点存在
        inTree.children[items[0]].inc(count)    # 更新该元素的次数
    else:
        inTree.children[items[0]] = treeNode(items[0],count,inTree) # 创建一个新的数据类型并将其作为一个子节点添加到树种
        if headerTabel[items[0]][1] == None:
            # 原来不存在该类别，更新头指针列表
            headerTabel[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTabel[items[0]][1],inTree.children[items[0]])
    if len(items) > 1:  # 迭代自身，直到全部元素添加至树种
        updateTree(items[1::],inTree.children[items[0]],headerTabel,count)

def updateHeader(nodeToTest,targetNode):
    # 说实话没太理解这个函数....
    # 书上的原画是：确保节点链接指向树中该元素项的每一个实例....
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    # 创建一个简单的数据集
    simpDat = [['r','z','h','j','p'],
               ['z','y','x','w','v','u','t','s'],
               ['z'],
               ['r','x','n','o','s'],
               ['y','r','x','z','q','t','p'],
               ['y','z','x','e','q','s','t','m']]
    return simpDat

def createInitSet(dataSet):
    # 初始化数据集，确保数据集符合要求
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def ascendTree(leafNode,prefixPath):
    # 迭代整棵树
    if leafNode.parent != None:
        # 如果给定元素的父节点不为空
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)
        
def findPrefixPath(basePat,treeNode):
    # 给定元素，查找条件模型基
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode,prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count    # 获取节点次数
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree,headerTable,minSup,preFix,freqItemList):
    # 递归查找频繁项集
    bigL = [v[0] for v in sorted(headerTable.items(),key=lambda p:str(p[1]))]    # 对头指针出现频率排序
    for basePat in bigL:
        newFreqSet = preFix.copy()          #加入频繁项列表
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)     #递归调用函数来创建基
        condPattBases = findPrefixPath(basePat,headerTable[basePat][1])
        myCondTree,myHead = createTree(condPattBases,minSup)    #将创建的条件基作为新的数据集添加到fp-tree
        print ('conditional tree for :',newFreqSet)
        if myHead != None:  #递归调用自身
            print ('conditional tree for :',newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)
    




