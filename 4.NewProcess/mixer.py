# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 20:05:53 2024

@author: DELL
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
#加载当前目录下的所有文件 返回X,y
import os
path='Features25/tractor'
tablenames=os.listdir(path)
train,_=train_test_split(tablenames, test_size=0.2)
valid,test=train_test_split(_,test_size=0.5)

columns:list
def sliding_channel_windows(tablenames):
    global columns
    X=np.empty(shape=(0,25,9))
    y=[]
    j=1
    for tablename in tablenames:
        temp=pd.read_excel(os.path.join('Features25/tractor',tablename))
        label=temp['tags'].tolist()
        temp=temp.drop(['lon','lat','tags','dir'],axis=1)
        columns=temp.columns
        temp=np.array(temp)
        data=np.expand_dims(temp,axis=2)
        #9通道滑窗
        for i in range(8):
            data=np.concatenate((data[:-1],np.expand_dims(temp[i+1:],axis=2)),axis=2)
        label=label[4:-4]
        X=np.concatenate((X,data))
        y+=label
    print(f'The {j}th tracjectory has processed!')
    j+=1        
    return X,y #numpy:Numbers*Features*Channels  list:label
#采用划分后的数据
X_train,y_train=sliding_channel_windows(train)
X_valid,y_valid=sliding_channel_windows(valid)
X_test,y_test=sliding_channel_windows(test)

from torchvision.io import read_image
# 自定义数据集
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import Lambda

class mydataset(Dataset):
    #data label均是列表形式
    def __init__(self,data,label,mode='valid'):
        #转换成张量
        #(n,25,9)->(n,9,25)
        self.data=torch.tensor(data)
        self.data=torch.permute(self.data,(0,2,1))
        self.label=torch.tensor(label)
        self.transforms = T.Compose([
            T.Normalize(mean=[0.485], std=[0.229])
        ])
        #One-hot Encode
        self.target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, y.clone().detach(), value=1))
    def __len__(self):#数据集划分成多少个batch时有__len__方法确定的
        return len(self.label)
    def __getitem__(self,idx):
        image=self.data[idx]
        label=self.label[idx]
        label=self.target_transform(label)
        return image.to('cuda').float(),label.to('cuda').float()

#validDataloader testDataLoader trainDataLoader
trainDataset=mydataset(X_train, y_train)
validDataset=mydataset(X_valid,y_valid)
testDataset=mydataset(X_test,y_test)

from torch.utils.data import DataLoader
trainDataLoader=DataLoader(trainDataset,batch_size=64,shuffle=True,num_workers=0)#单进程 注：在CPU上才能多进程
validDataLoader=DataLoader(validDataset,batch_size=64,shuffle=True,num_workers=0)#单进程
testDataLoader=DataLoader(testDataset,batch_size=64,shuffle=True,num_workers=0)#单进程

# a,b=next(iter(testDataLoader))


import torchvision
# from torchvision.datasets import CIFAR10
# traindata=CIFAR10('exampledata',train=True,download=True)
# testdata=CIFAR10('exampledata',train=False,download=True)
import torch
from torch import nn

class MixerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_mixing=nn.Sequential(
            nn.Linear(9,9),
            nn.GELU(),
            nn.Linear(9,9)
            )
        self.norm1=nn.LayerNorm(25)
        self.channel_mixing=nn.Sequential(
            nn.Linear(25,25),
            nn.GELU(),
            nn.Linear(25,25)
            )
        self.norm2=nn.LayerNorm(9)
        
    def forward(self,x):
        y1=self.norm1(x)
        y1=torch.permute(y1,(0,2,1))
        y1=self.token_mixing(y1)
        y1=torch.permute(y1,(0,2,1))
        xx=x+y1
        
        y2=torch.permute(xx,(0,2,1))
        y2=self.norm2(y2)
        y2=torch.permute(y2,(0,2,1))
        y2=self.channel_mixing(y2)
        return xx+y2
from typing import OrderedDict
class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.startblock:nn.Module
        
        blocks=OrderedDict()
        for i in range(3):
            blocks['MixerBlock'+str(i)]=MixerBlock()
        self.middleblock=nn.Sequential(blocks)
        self.endblock=nn.Sequential(
            nn.LayerNorm([9,25]),
            nn.AvgPool1d(25),
            nn.Flatten(),
            nn.Linear(9, 2),
            nn.Softmax(dim=1)
            )
    def forward(self,x):#x.shape 1*9*25  patches*channels
        y=self.middleblock(x)
        # y=y.mean(dim=2)#全局平均池化
        y=self.endblock(y)
        return y


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

mlp=Mixer().to('cuda')#模型的整个执行过程在GPU上进行 
# print(mlp)
#输出模型参数总量
from pytorch_model_summary import summary
tmp = torch.randn(1,9,25).to('cuda')
print(summary(mlp,tmp, show_input=False, show_hierarchical=False))

#--------------------------训练--------------------------------------#
#模型训练
from torch.optim import SGD
loss_fun=nn.CrossEntropyLoss()
optimizer=SGD(mlp.parameters(),lr=0.1)

#模型评估
epoches=10
train_loss=[]
valid_acc=[]
for i in range(epoches):
    print(f'The {i+1}st epoch Begin.')
    train_loss.append(train_eval(trainDataLoader, mlp, loss_fun, optimizer))
    valid_acc.append(valid_eval(validDataLoader, mlp, loss_fun, optimizer))

plotcurve(train_loss,'Train Loss')
plotcurve(valid_acc,'Valid Accuracy')
c_matrix,report=valid_eval(testDataLoader, mlp, loss_fun, optimizer,mode='test')
print(f'测试集混淆矩阵:\n{c_matrix}')
print(f'测试集详细结果:\n{report}')

# train_eval(testDataLoader, mlp, loss_fun, optimizer)




