# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:40:26 2017

@author: 凯风
"""

import numpy as np
from numpy import linalg as la

def loadExData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]

def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def eculidSim(inA,inB):
    # 计算欧氏距离，然后把欧氏距离折算成0-1之间
    return 1.0/(1.0+la.norm(inA-inB))

def pearsSim(inA,inB):
    # 计算皮尔逊距离，也是把距离折算成0-1之间
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    # 计算余弦相似度，也是0-1
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5*(num/denom)


def standEst(dataMat,user,simMeas,item):
    # 基于物品相似度（也就是基于n的）的推荐，这个函数主要是计算相似度
    # 接收：数据集、用户数据m、计算相似度的方式、商品集（对应用户未评价的）
    n = np.shape(dataMat)[1]    # 获取n，商品数
    simTotal = 0.0              #初始化估计评分的变量
    ratSimTotal =0.0
    for j in range(n):  # 迭代全部商品
        userRating = dataMat[user,j]      # 该用户对商品的评分
        if userRating == 0: continue      # 跳过未评分的
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]   
        # 关于这个overlap，一个是已经评价的商品item，获得所有用户的对它的评价列，一个是该用户未评价的商品j，获得所有用户对它的评价列
        # 然后分别获得对两个商品均评价的bool数组
        # 然后获得两个bool数组中都是True的，其实就是找对这两个商品均评价了的用户清单
        if len(overLap) == 0:   #  如果为0则终止本次
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])  # 计算相似度
        print ('the %d and %d similarity is :%f'%(item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal == 0 :
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas = cosSim,estMethod=standEst):
    # 具体的推荐系统
    # 接收：数据集、用户数据、要推荐的数量、计算相似度的方式、估计评分方法
    unratedItems = np.nonzero(dataMat[user,:].A==0)[1]      # 获取用户未评价的商品ID
    if len(unratedItems) ==0:   # 如果全部都评价了就算了
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:   # 迭代全部未评价的商品
        estimatedScore = estMethod(dataMat,user,simMeas,item)   # 调用方法计算可能的评分
        itemScores.append((item,estimatedScore))                # 添加到列表中
    return sorted(itemScores,key=lambda jj:jj[1] , reverse=True)[:N]    # 返回前N个未评价的最高分的


def svdEst(dataMat,user,simMeas,item):
    # 估计评分方法之SVD，和之前standEst接收的肯定是一样的
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)                    # 获得svd的三个矩阵
    Sig4 = np.mat(np.eye(4) * Sigma[:4])            # 构建对角矩阵,对角线上的值是奇异值，取前四个
    xformedItems = dataMat.T * U[:,:4] * Sig4.I     # 压缩数据集
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)    # 相似度在低纬度进行计算
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def printMat(inMat,thresh=0.8):
    # 打印矩阵，thresh是个阈值，只要在0-1之间就好
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print (1,)  # 这里可能有些问题
            else:
                print (0,)
        print ('')          # 这里可能也有些问题

def imgCompress(numSV=3,thresh=0.8):
    # 基于给定的奇异值数目来压缩图片
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myDat = np.mat(myl) 
    print ("****original matrix******")
    printMat(myDat,thresh)                      # 打印原矩阵
    U,Sigma,VT = la.svd(myDat)                  # 求解奇异值
    SigRecon = np.mat(np.zeros((numSV,numSV)))  # 创建一个符合奇异值数目的0矩阵
    for k in range(numSV):                      # 创建奇异值对角矩阵
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV] * SigRecon * VT[:numSV,:]    # 重塑特征,这里的问题就是重塑之后的值，不是0,1了，那thresh还满足么。。。
    print ("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat,thresh)                 # 打印压缩后的矩阵    