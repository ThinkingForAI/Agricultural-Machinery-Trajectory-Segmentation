# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:49:00 2024

@author: DELL
"""
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
def sliding_windows(path,file,tablenames,num):
    # X=np.empty(shape=(0,9,25)) #slow
    # y=[]
    # print('||',end='')
    # for table in tablenames:
    #     tmp=pd.read_excel(os.path.join(path,file,table))
    #     label=tmp['tags'].tolist()
    #     tmp=tmp.drop(['lon','lat','tags','dir'],axis=1).values
    #     for i in range(len(tmp)-8):
    #         slide=np.expand_dims(tmp[i:i+9],axis=0)
    #         X=np.concatenate((X,slide),axis=0)
    #     label=label[4:-4]  
    #     y+=label
    #     print(end='>>')
    # print('||')
    
    X=np.empty(shape=(0,25,num)) #fast
    y=[]
    print('START ',end='')
    for table in tablenames:
        temp=pd.read_excel(os.path.join(path,file,table))
        label=temp['tags'].tolist()
        temp=temp.drop(['lon','lat','tags','dir'],axis=1).values
        data=np.expand_dims(temp,axis=2)
        #num通道滑窗
        for i in range(num-1):
            data=np.concatenate((data[:-1],np.expand_dims(temp[i+1:],axis=2)),axis=2)
        label=label[int((num-1)/2):int(-(num-1)/2)]
        X=np.concatenate((X,data))
        y+=label
        print(end='>')
    print(' END')
    return X,y

def data_to_tensor(path,tensorfile,windownums):
    for file in os.listdir(path):
        tablenames=os.listdir(os.path.join(path, file))
        train,_=train_test_split(tablenames, test_size=0.3)
        valid,test=train_test_split(_, test_size=0.3)
        X_train,y_train=sliding_windows(path,file,train,windownums)
        print(file+str(len(train))+' train trajectory has all completed')
        X_valid,y_valid=sliding_windows(path,file,valid,windownums)
        print(file+str(len(valid))+' valid trajectory has all completed')
        X_test,y_test=sliding_windows(path,file,test,windownums)
        print(file+str(len(test))+' test trajectory has all completed')
        torch.save([torch.tensor(X_train), torch.tensor(y_train)],os.path.join(tensorfile,file+str(windownums)+'train'))
        torch.save([torch.tensor(X_valid), torch.tensor(y_valid)],os.path.join(tensorfile,file+str(windownums)+'valid'))
        torch.save([torch.tensor(X_test), torch.tensor(y_test)],os.path.join(tensorfile,file+str(windownums)+'test'))
    return 

if __name__=='__main__':
    try:
        os.makedirs('feature_tensor',exist_ok=False)
    except FileExistsError:
        print('feature_tensor has existed')
    data_to_tensor('Features25', 'feature_tensor', 9)#9 25 49 81 121
    # data_to_tensor('Features25', 'feature_tensor', 25)#9 25 49 81 121
    # data_to_tensor('Features25', 'feature_tensor', 49)#9 25 49 81 121
    