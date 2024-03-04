# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 09:21:12 2024

@author: DELL
"""

## -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#处理后轨迹全部读入
featurefile='tractor_25'
trajectory=os.listdir(featurefile)

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#每条轨迹分别聚类
accuracy=[]
precision=[]
recall=[]
f1_score=[]
for trace in trajectory:
    data=pd.read_excel(featurefile+'/'+trace)
    scaler=StandardScaler()
    X_train=np.array(data.drop(['tag','lon','lat'],axis=1))
    y_true=np.array(data['tag'])
    #标准化
    scaler.fit_transform(X_train)
    
    #PCA主成分分析
    pca=PCA()
    pca.fit(X_train)
    tmp=np.cumsum(pca.explained_variance_ratio_)
    # plt.plot(tmp,'r--')
    # plt.title('Accumulated Variance Ratio')
    # plt.show()
    for i in range(len(tmp)):
        if tmp[i]>0.85:
            break
    pca=PCA(n_components=i+1)
    X_train=pca.fit_transform(X_train)
    #KMeans聚类
    clustering=KMeans(n_clusters=2)
    clustering.fit(X_train)
    y_pred=clustering.labels_
    accuracy.append(accuracy_score(y_true, y_pred))
    precision.append(precision_score(y_true, y_pred))
    recall.append(recall_score(y_true, y_pred))

plt.plot(accuracy,'b-',label='Accuracy')
plt.plot(precision,'r--',label='Precision')
plt.plot(recall,'y-.',label='Recall')
plt.title('Metrics of Every Trajectory')
plt.legend()
plt.show()
