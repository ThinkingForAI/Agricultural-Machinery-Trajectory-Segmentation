# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 00:21:55 2024

@author: DELL
"""
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset,DataLoader
import os
datadict1={}
tablenames=os.listdir('CleanData/corn')
for tablename in tablenames:
    filepath=os.path.join('CleanData/corn',tablename)
    data=pd.read_excel(filepath,index_col=0)
    data.reset_index(drop=True,inplace=True)
    y=data['tags'][3:].values.reshape(-1,1)
    X=data.drop(['tags','time'],axis=1).values
    
    #sliding window
    slide=X[:4].flatten().reshape(1,-1)
    for i in range(5,len(data)+1):
        temp=X[i-4:i].flatten().reshape(1,-1)
        slide=np.concatenate((slide,temp))
        
    slide=pd.DataFrame(slide)
    slide['tags']=y
    datadict1[tablename]=slide

#-----------------------------------处理数据----------------------------------#
#划分
from sklearn.model_selection import train_test_split
#96:12:12
def dataconcat(datadict,datakeys):
    data=pd.DataFrame()
    for key in datakeys:
        data=pd.concat((data,datadict[key]))
    return data
def split(datadict):
    keys=list(datadict.keys())
    train,valid=train_test_split(keys,test_size=0.2,random_state=14)
    valid,test=train_test_split(valid,test_size=0.5,random_state=14)
    traindata=dataconcat(datadict, train)
    validdata=dataconcat(datadict, valid)
    testdata=dataconcat(datadict, test)
    return traindata,validdata,testdata

traindata,validdata,testdata=split(datadict1)

#标准化
from sklearn.preprocessing import StandardScaler
def normal(traindata,validdata,testdata):
    trainX,trainy=traindata.drop(['tags'],axis=1),traindata['tags']
    validX,validy=validdata.drop(['tags'],axis=1),validdata['tags']
    testX,testy=testdata.drop(['tags'],axis=1),testdata['tags']   
    std=StandardScaler()
    std.fit(trainX)
    trainX,validX,testX=std.transform(trainX),std.transform(validX),std.transform(testX)
    return trainX,trainy.values,validX,validy.values,testX,testy.values
    
trainX,trainy,validX,validy,testX,testy=normal(traindata,validdata,testdata)





#-----------------------------------训练模型----------------------------------#
import sklearn
class mydata(Dataset):
    def __init__(self,X,y):
        #numpy
        self.data=torch.tensor(X).to('cuda').float()
        self.label=torch.tensor(y).to('cuda').float()#标签也需是二维
    #继承时需要重写方法
    def __len__(self):
        return len(self.label)
    def __getitem__(self,idx):
        data=self.data[idx]
        label=self.label[idx]
        if self.label[idx]==torch.tensor(0):
            label=torch.tensor([1,0]).to('cuda').float()
        if self.label[idx]==torch.tensor(1):
            label=torch.tensor([0,1]).to('cuda').float()
        return data,label
    
trainset=mydata(trainX,trainy)
validset=mydata(validX,validy)
testset=mydata(testX,testy)
#执行过程中numpy被转换为tensor
trainloader=DataLoader(trainset,batch_size=64,shuffle=False,)
validloader=DataLoader(validset,batch_size=64,shuffle=False,)
testloader=DataLoader(testset,batch_size=64,shuffle=False,)


#-----------------------------------模型训练----------------------------------#
#代码详细解读
class mynet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(
            torch.nn.Linear(28, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 2),
            torch.nn.Softmax(dim=1)
            )        
    def forward(self,x):
        return self.layers(x)
def train_eval(DataLoader,model,loss_fun,optimizer):
    size=len(DataLoader)#每个batch的大小为64，size为batch的总量，对valid一共1773个batch
    model.train()#说明：dropout和BN在训练和验证时的作用不同
    avg_loss=0
    for batch,(X,y) in enumerate(DataLoader):#仅仅只是增加了序号
        #前向计算
        pred=model(X)#多个输出->tensor model(X)等价于model.forward(X)
        loss=loss_fun(pred,y)#损失计算
        #后向传播
        loss.backward()# torch.Tensor类具有一个属性 requires_grad 用以表示该tensor是否需要计算梯度
        optimizer.step()
        optimizer.zero_grad()
        loss, current = loss.item(), (batch + 1) * len(X)
        avg_loss+=loss
    avg_loss/=size
    print(f"Train Error: \n\t\t\t\t  Avg loss: {avg_loss:>8f} \n")
    return avg_loss/size
      
from sklearn.metrics import confusion_matrix,classification_report
def valid_eval(DataLoader,model,loss_fun,optimizer,mode='validation'):
    model.eval()#关闭dropout BN采用随机方式
    size=len(DataLoader.dataset)#数据总量
    num_batches = len(DataLoader)#batches总数
    valid_loss,valid_acc=0,0 
    #验证时取消梯度计算的含义？
    with torch.no_grad():
    # torch.no_grad() 停止自动微分的计算，计算训练，节省存储，不影响Dropout和BN的行为
        if mode=='test':
            y_pred=[]
            y_true=[]
        for X,y in DataLoader:
            pred=model(X)
            valid_loss+=loss_fun(pred,y).item()
            valid_acc+=(pred.argmax(1)==y.argmax(1)).float().sum().item()
            if mode=='test':
                y_pred+=pred.argmax(1).to('cpu')
                y_true+=y.argmax(1).to('cpu')
        valid_loss/=num_batches
        valid_acc/=size
    if mode=='validation':
        print(f"Test Error: \n Accuracy: {(100*valid_acc):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
        return valid_acc
    else:
        return confusion_matrix(y_true,y_pred),classification_report(y_true, y_pred)
def plotcurve(data,name):
    if 'Loss' in name:
        plt.plot(data,'r')
    else:
        plt.plot(data)
    plt.title(name+' Curve')
    plt.grid()
    plt.ylabel(name)
    plt.xlabel(name)
    plt.show()



#-----------------------------------模型初始化----------------------------------#
mlp=mynet().to('cuda')#模型的整个执行过程在GPU上进行 
# #输出模型参数总量
# from pytorch_model_summary import summary
# tmp = torch.randn(1, 1, 32, 32).to('cuda')
# print(summary(mlp,tmp, show_input=False, show_hierarchical=False))

from torch.optim import SGD
#模型训练
loss_fun=torch.nn.CrossEntropyLoss()
optimizer=SGD(mlp.parameters(),lr=0.1)
#模型评估
epoches=10

train_loss=[]
valid_acc=[]
for i in range(epoches):
    print(f'The {i+1}st epoch Begin.')
    train_loss.append(train_eval(trainloader, mlp, loss_fun, optimizer))
    valid_acc.append(valid_eval(validloader, mlp, loss_fun, optimizer))

plotcurve(train_loss,'Train Loss')
plotcurve(valid_acc,'Valid Accuracy')
c_matrix,report=valid_eval(testloader, mlp, loss_fun, optimizer,mode='test')
print(f'测试集混淆矩阵:\n{c_matrix}')
print(f'测试集详细结果:\n{report}')