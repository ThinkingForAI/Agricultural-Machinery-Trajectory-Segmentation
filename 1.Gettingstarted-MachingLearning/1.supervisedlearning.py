# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# #原始数据全部读入
# data=pd.read_excel('tractor/tractor_1.xlsx')
# #数据清洗
# #异常值判断
# for key in ['longitude','latitude','height','dir','speed']:
#     plt.boxplot(data[key])
#     plt.title(key)
#     plt.show()
#标签统计
# stat=data.groupby('tags').describe()[('longitude','count')]
# plt.bar(['Field','Road'],stat.values_a,width=0.3)
# plt.title('Model Distribution')
# plt.show()
# #缺失值判断
# data.isnull().sum()
# #重复值处理
# data.drop_duplicates('time')
# data.drop_duplicates(['longitude','latitude','speed'])

# 数据清洗+数据预处理并保存结果
# clean_and_featureExtract()

#处理后轨迹全部读入
featurefile='tractor_25'
trajectory=os.listdir(featurefile)
#构造测试集 一共60条轨迹选取10条轨迹充当测试集
import random
random.seed(1)
test_trajectory=random.sample(trajectory,10)
for i in test_trajectory:
    trajectory.remove(i)#剩余的轨迹充当训练集
#10折交叉验证
from sklearn.model_selection import KFold,train_test_split
kf=KFold(n_splits=10,shuffle=False,random_state=None)
keys=['DT','Bayes','LDA','SVM']
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time

data={}
for trace in trajectory:
    data[trace]=pd.read_excel(featurefile+'/'+trace)
    
accuracy=[]
recall=[]
precision=[]
times=[]
scaler=StandardScaler()
#平衡标签
flag=0
def balance_labels(X_train,y_train):
    global flag
    if flag!=2:
        plt.bar(['Field','Road'],[sum(y_train==1),sum(y_train==0)],width=0.3)
        plt.title('Before Balance Label Distribution')
        plt.show()
    X_field=X_train[y_train==1,:]
    y_field=y_train[y_train==1]
    X_road=X_train[y_train==0,:]
    y_road=y_train[y_train==0]
    #打乱Dataset [X,y]
    X_field,_1,y_field,_2=train_test_split(X_field,y_field,train_size=sum(y_train==0)/sum(y_train==1))
    X_train=np.concatenate((X_field,X_road))
    y_train=np.concatenate((y_field,y_road))
    #打乱Dataset [X,y]
    X_train,_1,y_train,_2=train_test_split(X_train,y_train,train_size=9999/10000,shuffle=True)
    if flag!=2:
        plt.bar(['Field','Road'],[sum(y_train==1),sum(y_train==0)],width=0.3)
        plt.title('After Balance Label Distribution')
        plt.show()
    flag+=1
    return X_train,y_train

for train_index, valid_index in kf.split(trajectory):
    train=pd.DataFrame()
    for i in train_index:
        train=pd.concat((train,data[trajectory[i]]))
    valid=pd.DataFrame()
    for j in valid_index:
        valid=pd.concat((valid,data[trajectory[j]]))
    X_train=np.array(train.drop(['tag','lon','lat'],axis=1))
    y_train=np.array(train['tag'])
    #平衡标签训练集
    X_train,y_train=balance_labels(X_train, y_train)
  
    X_valid=np.array(valid.drop(['tag','lon','lat'],axis=1))
    y_valid=valid['tag']
    #平衡标签验证集
    X_valid,y_valid=balance_labels(X_valid, y_valid) 
    
    #标准化
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_valid=scaler.transform(X_valid)
    # break
    
    values_a=[]
    values_r=[]
    values_t=[]
    values_p=[]
    
    #模型训练
    
    #DT
    t0=time.time()
    tree=DecisionTreeClassifier()
    tree.fit(X_train,y_train)
    y_pred_tree=tree.predict(X_valid)
    values_a.append(accuracy_score(y_valid, y_pred_tree))
    values_r.append(recall_score(y_valid, y_pred_tree))
    values_p.append(precision_score(y_valid, y_pred_tree))
    t1=time.time() 
    values_t.append(t1-t0)
    
    #NaiveBayes
    t0=time.time()
    bayes=GaussianNB()
    bayes.fit(X_train,y_train)
    y_pred_bayes=bayes.predict(X_valid)
    values_a.append(accuracy_score(y_valid, y_pred_bayes))   
    values_r.append(recall_score(y_valid, y_pred_bayes))
    values_p.append(precision_score(y_valid, y_pred_bayes))
    t1=time.time() 
    values_t.append(t1-t0)
    
    #LDA
    t0=time.time()
    lda=LinearDiscriminantAnalysis()
    lda.fit(X_train,y_train)
    y_pred_lda=lda.predict(X_valid)
    values_a.append(accuracy_score(y_valid, y_pred_lda))   
    values_r.append(recall_score(y_valid, y_pred_lda))
    values_p.append(precision_score(y_valid, y_pred_lda))
    t1=time.time() 
    values_t.append(t1-t0)
    

    #SVM
    t0=time.time()
    svc=SVC(kernel='poly',max_iter=300)
    svc.fit(X_train, y_train)
    y_pred_svc=svc.predict(X_valid)

    values_a.append(accuracy_score(y_valid, y_pred_svc))
    values_r.append(recall_score(y_valid, y_pred_svc))
    values_p.append(precision_score(y_valid, y_pred_svc))
    t1=time.time() 
    values_t.append(t1-t0)
    
    accuracy.append(values_a)
    recall.append(values_r)
    times.append(values_t)
    precision.append(values_p)
    
#模型评估
accuracy=np.array(accuracy)
recall=np.array(recall)
times=np.array(times)
precision=np.array(precision)

#选择最佳模型 
linetype=['b-.','g--x','y-.+','r:*']
for i in range(len(keys)):
    plt.plot(accuracy[:,i],linetype[i],label=keys[i])
plt.title('Validation Accuracy Metrics')
plt.legend()
plt.show()

for i in range(len(keys)):
    plt.plot(recall[:,i],linetype[i],label=keys[i])
plt.title('Validation Recall Metrics')
plt.legend()
plt.show()

for i in range(len(keys)):
    plt.plot(precision[:,i],linetype[i],label=keys[i])
plt.title('Validation Precision Metrics')
plt.legend()
plt.show()

for i in range(len(keys)):
    plt.plot(times[:,i],linetype[i],label=keys[i])
plt.title('Validation Time Metrics')
plt.legend()
plt.show()


#训练轨迹读入
train=pd.DataFrame()
for i in trajectory:
    train=pd.concat((train,data[i]))
X_train=np.array(train.drop(['tag','lon','lat'],axis=1))
y_train=np.array(train['tag'])  
#平衡标签训练集
X_train,y_train=balance_labels(X_train, y_train)

#测试轨迹读入
test=pd.DataFrame()
for i in test_trajectory:
    test=pd.concat((test,pd.read_excel(featurefile+'/'+i)))
X_test=np.array(test.drop(['tag','lon','lat'],axis=1))
y_test=np.array(test['tag'])
#平衡标签验证集
X_test,y_test=balance_labels(X_test, y_test)


#确定最佳模型最佳参数
bestmodel=LinearDiscriminantAnalysis()
bestmodel1=LinearDiscriminantAnalysis()
#不做标准化
bestmodel1.fit(X_train,y_train)
best_pred_noscaler=bestmodel1.predict(X_test)

#标准化
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
bestmodel.fit(X_train,y_train)  
X_test1=scaler.transform(X_test)
best_pred=bestmodel.predict(X_test1)


#最终测试集准确率
from sklearn.metrics import classification_report
print("不做标准化")
print(classification_report(y_test, best_pred_noscaler))
print("标准化")
print(classification_report(y_test, best_pred))


# print("不做标签均衡处理")
# print(classification_report(y_test, best_pred))




