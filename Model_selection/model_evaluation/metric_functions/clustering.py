# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:24:59 2017

@author: 凯风
"""

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,\
                            adjusted_mutual_info_score,\
                            normalized_mutual_info_score,\
                            homogeneity_score,\
                            completeness_score,\
                            homogeneity_completeness_v_measure,\
                            fowlkes_mallows_score,\
                            silhouette_score,\
                            calinski_harabaz_score,\
                            v_measure_score
# 案例一
iris_dataset = load_iris()
X = iris_dataset.data
Y = iris_dataset.target                  
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
# 案例二
label_true = [0,1,2,0,3,2,3,1]
label_pred = [0,2,2,0,3,1,2,1]

'''
    聚类指标：
        adjusted_rand_score
            调整兰德指数，区间[-1,1]，ARI = (RI - E(RI))/(max(RI) - E(RI))
            优点：不做什么假设，适用性广
            缺点：需要知道真实值
        adjusted_mutual_info_score
            调整互信息，区间[-1,1],AMI = (MI - E(MI)) / (max(H(U),H(V)) - E(MI))
            优点：不做什么假设，
            缺点：需要真实值
        normalized_mutual_info_score
            标准互信息，区间[0,1], NMI = MI(U,V) / (H(U)*H(V))^0.5
            缺点：没有调整的机会
        homogeneity_score
            均一性，公式什么的在cv可用参数里面有
        completeness_score
            完整性，同上
        v_measure_score
            V度量，上面两个的加权
        homogeneity_completeness_v_measure
            一次性计算：同质性、完整性、V度量
        fowlkes_mallows_score
            FMI = TP/((TP + FP)*(TP + FN))^0.5
            区间在[0,1]
        silhouette_score
            轮廓系数，区间[-1,1], s = (b-a)/max(a,b)
            区间在[-1,1]
        calinski_harabaz_score
            暂时先不搞了。。。
'''

adjusted_rand_score(label_true,label_pred)
adjusted_mutual_info_score(label_true,label_pred)
normalized_mutual_info_score(label_true,label_pred)
homogeneity_score(label_true,label_pred)
completeness_score(label_true,label_pred)
v_measure_score(label_true,label_pred)
homogeneity_completeness_v_measure(label_true,label_pred)
fowlkes_mallows_score(label_true,label_pred, sparse=False)

silhouette_score(X,labels,metric = 'euclidean', sample_size=None, random_state=None)
calinski_harabaz_score(X,labels)

'''
    labels_true
    labels_pred
    sparse
    sample_size
    random_state
'''