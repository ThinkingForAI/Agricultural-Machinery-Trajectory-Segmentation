# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:32:06 2024

@author: DELL
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch

#检查GPU是否可用
torch.cuda.is_available()

#张量初始化 dtype=? 指明数据类型 和存储
#列表 
torch.tensor([1,2],dtype=float,device='cuda')
#numpy数组
a=np.array([[1,2],[2,3]])

b=torch.tensor(a) #或者 torch.from_numpy(a)
c=b.numpy()#张量自带的numpy()方法共享存储空间

#存储位置转移
c.to('cuda')#默认cpu 当前变量的副本

#原地操作
b.add_(1)
#单个变量取值
b[1,0].item()

#列表化
b.tolist()
b.flatten().tolist() #flatten(1(start dim),-1(end dim))#默认不从第0维开始



#加载当前目录下的所有文件 返回X,y
import os
#path='tractor'
def wrap(path):
    excel_names=os.listdir(path)
    print(f'There ara {len(excel_names)} trajectory')
    X=[]
    y=[]
    for name in excel_names:
        curpath=os.path.join(path,name)
        data=pd.read_excel(curpath)
        tmp=data.drop(['tag','lon','lat'],axis=1).astype(float).values.tolist()
        #连续四点合并 最后四点除去
        for i in range(len(tmp)-4):
            tmp[i]=tmp[i]+tmp[i+1]+tmp[i+2]+tmp[i+3]
        del(tmp[i+1:])
        X+=tmp
        y+=data['tag'].tolist()[:-4]
    return X,y
#采用划分后的数据
X_test,y_test=wrap('tractor/test')
X_valid,y_valid=wrap('tractor/val')
X_train,y_train=wrap('tractor/train')

def plotgraph(x,y,shape):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 2, 2
    import random
    labels_map=['road','field']
    for i in range(1, cols * rows + 1):
        idx=random.randint(0,len(x)-1)
        img=np.array(x[idx]).reshape(shape)
        label=y[idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img, cmap="gray")
    plt.show() 

#绘制四点图像
plotgraph(X_train,y_train,(10,10))


#CV公用数据集的使用
from torchvision import datasets
#mnist=datasets.FashionMNIST('root',train=True,transform=?,target_transform=?,download=True)
# 返回结果是torch.utils.data.Dataset的类型
# 并行执行时常使用 torch.utils.data.DataLoader


from torchvision.io import read_image
# 自定义数据集
from torch.utils.data import Dataset
#标准定义
class mystandarddataset(Dataset):
    def __init__(self,label_file,img_dir,transform=None,target_transform=None):
        #label_file csv文件的格式为 文件名,标签
        self.img_labels=pd.read_csv(label_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
    def __len__(self):
        return len(self.img_dir)
    
    def __getitem__(self,idx):
        #给出标签CSV文件的索引，返回对应图片的数据 和 标签 (张量形式)
        img_path=os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
        image=read_image(img_path)#读入的图片就是tensor 且已按三通道分开
        label=self.img_labels.iloc[idx,1]
        if self.transform:
            image=self.transform(image)
        if self.target_transform:
            label=self.target_transform(label) 
        return image,label

import torchvision.transforms as T

from torchvision.transforms import Lambda

class mydataset(Dataset):
    #data label均是列表形式
    def __init__(self,data,label,mode='valid'):
        #转换成张量
        #(n,100)->(n,1,10,10)
        #通道增加
        data=np.array(data).reshape(np.array(data).shape[0],1,10,10)
        self.data=torch.tensor(data)
        self.label=torch.tensor(label)
        
        self.transforms=T.Compose([
                T.Normalize(mean=self.data.mean(),std=self.data.std())#标准化
            ])
        #One-hot Encode
        self.target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, y.clone().detach(), value=1))
        #消除警告 torch.tensor(y)->y.clone().detach()(因为y是tensor)
        
        #     #torchvision.transforms.compose的用法？
        #     self.transforms = T.Compose([
        #         T.Resize((10,10)),                 # 图像大小修改
        #         T.ToTensor(),            # RGB数据的格式转换和标准化 HWC => CHW
        #         T.Normalize(mean=[0.485], std=[0.229])   # 图像归一化
        #     ])
    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        image=self.data[idx]
        label=self.label[idx]
        image=self.transforms(image)
        label=self.target_transform(label)
        return image.to('cuda').float(),label.to('cuda').float()

trainDataset=mydataset(X_train, y_train)
validDataset=mydataset(X_valid,y_valid)
testDataset=mydataset(X_test,y_test)


from torch.utils.data import DataLoader
trainDataLoader=DataLoader(trainDataset,batch_size=64,shuffle=True,num_workers=0)#单进程
validDataLoader=DataLoader(validDataset,batch_size=64,shuffle=True,num_workers=0)#单进程
testDataLoader=DataLoader(testDataset,batch_size=64,shuffle=True,num_workers=0)#单进程

#如何从DataLoader读出数据？
#next+iter方法
data,labels=next(iter(validDataLoader))#一次返回一个batch 结构A*B*C

#收缩维度 tensor.squeeze()
figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 2
import random
labels_map=['road','field']
for i in range(1, cols * rows + 1):
    idx=random.randint(0,len(data)-1)
    img=data[idx].squeeze()
    label=labels[idx].argmax(0)#在第0维上的最大值
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.to('cpu'), cmap="gray")
plt.show() 



#validDataloader testDataLoader trainDataLoader
from torch import nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.layers=nn.Sequential(
            nn.Linear(10*10, 30),
            nn.Sigmoid(),
            nn.Linear(30, 2),
            nn.Softmax(dim=1)
            )
    def forward(self,x):
        x=self.flatten(x)
        prob=self.layers(x)
        return prob

mlp=MLP().to('cuda')#模型的整个执行过程在GPU上进行 但数据的存储在CPU上
# print(mlp)

#遍历模型初始化的所有参数
for name,params in mlp.named_parameters():
    print(f'Current Layer:{name},Parameters{params}')
    break


from torch.optim import SGD

#模型训练
loss_fun=nn.CrossEntropyLoss()
optimizer=SGD(mlp.parameters(),lr=0.5)

# #梯度更新三步
# 1)optimizer.zero_grad() 每个batch训练后梯度归零（默认自动增加）
# 2)loss.backward() 后向传播,计算每层梯度
# 3）optimizer.step() 更新参数

#模型评估
def train_eval(DataLoader,model,loss_fun,optimizer):
    size=len(DataLoader)#每个batch的大小为64，size为batch的总量，对valid一共1773个batch
    
    model.train()#说明：dropout和BN在训练和验证时的作用不同
    
    # list(enumerate(validDataLoader))[0][1][0].shape
    # Out: torch.Size([64, 1, 10, 10])\
    #一个batch的训练是同时进行的
    for batch,(X,y) in enumerate(validDataLoader):#仅仅只是增加了序号
        #前向计算
        pred=model(X)#多个输出->tensor model(X)等价于model.forward(X)
        loss=loss_fun(pred,y)#损失计算
        
        #如何追踪梯度？？？？？
        #解释
        # print('Before loss.backward(),Before optimizer.step():')
        # for name,params in mlp.named_parameters():
        #     print(f'Current Layer:{name}')
        #     print(f'Current Params:{params}')
        #     print(f'Update Grad or not:{params.requires_grad}')
        #     print(f'Current Layer Grad:{params.grad}')
        #     break
    
        #后向传播
        loss.backward()# torch.Tensor类具有一个属性 requires_grad 用以表示该tensor是否需要计算梯度
        # print('After loss.backward(),Before optimizer.step():')
        # for name,params in mlp.named_parameters():
        #     print(f'Current Layer:{name}')
        #     print(f'Current Params:{params}')
        #     print(f'Update Grad or not:{params.requires_grad}')
        #     print(f'Current Layer Grad:{params.grad}')
        #     break
    
        optimizer.step()
        # print('After loss.backward(),After optimizer.step():')
        # for name,params in mlp.named_parameters():
        #     print(f'Current Layer:{name}')
        #     print(f'Current Params:{params}')
        #     print(f'Update Grad or not:{params.requires_grad}')
        #     print(f'Current Layer Grad:{params.grad}')
        #     break
    
        optimizer.zero_grad()
        # print('After optimizer.zero_grad():')
        # for name,params in mlp.named_parameters():
        #     print(f'Current Layer:{name}')
        #     print(f'Current Params:{params}')
        #     print(f'Update Grad or not:{params.requires_grad}')
        #     print(f'Current Layer Grad:{params.grad}')
        #     break
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
def valid_eval(DataLoader,model,loss_fun,optimizer):
    model.eval()#关闭dropout BN采用随机方式
    size=len(DataLoader.dataset)#数据总量
    num_batches = len(DataLoader)#batches总数
    valid_loss,valid_acc=0,0 
    
    #验证时取消梯度计算的含义？
    with torch.no_grad():
    # torch.no_grad() 停止自动微分的计算，计算训练，节省存储，不影响Dropout和BN的行为
    
        for X,y in DataLoader:
            pred=model(X)
            valid_loss+=loss_fun(pred,y).item()
            valid_acc+=(pred.argmax(1)==y.argmax(1)).float().sum().item()
        valid_loss/=num_batches
        valid_acc/=size
    print(f"Test Error: \n Accuracy: {(100*valid_acc):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
    

# train_eval(trainDataLoader, mlp, loss_fun, optimizer)
train_eval(validDataLoader, mlp, loss_fun, optimizer)
# train_eval(testDataLoader, mlp, loss_fun, optimizer)
valid_eval(validDataLoader, mlp, loss_fun, optimizer)

#模型保存

#经典模型调用
import torchvision.models as models
# model = models.vgg16(weights='IMAGENET1K_V1')

#参数保存
torch.save(mlp.state_dict(),'mlp_parameters.pth')
#参数加载
mlp1=MLP()
mlp1.load_state_dict(torch.load('mlp_parameters.pth'))
mlp1.eval()


#模型保存
torch.save(mlp,'mlp.pth')
#模型加载
mlp2 = torch.load('mlp.pth')

    
    
    
    
    
    

