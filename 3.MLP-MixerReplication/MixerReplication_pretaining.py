import torch
import torchvision
import os
import sys
from torchvision.datasets import CIFAR100
train=CIFAR100('../MixerReplication',train=True,download=True)
test=CIFAR100('../MixerReplication',train=False,download=True)

Xtrain,ytrain=train.data,train.targets
Xtest,ytest=test.data,test.targets
types=train.classes
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#show one picture
# plt.imshow(Xtrain[0])
# plt.title(types[ytrain[0]])

#MLP-mixer
import torch
from torch import nn
SIZE=32
P=4*4
CHANNELS=192
PATCHES=int((SIZE/P)**2)
#可调隐藏宽度DS DC
DS=96
DC=768

class TokenMixer(nn.Module):
    def __init__(self):
        global PATCHES,DS,CHANNELS
        super().__init__()
        self.tokenblock=nn.Sequential(
            nn.Linear(PATCHES,DS),
            nn.BatchNorm1d(CHANNELS),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(DS,PATCHES),
            )
            
    def forward(self,x):
        return self.tokenblock(x)
    
class ChannelMixer(nn.Module):
    def __init__(self):
        global CHANNELS,DC,PATCHES
        super().__init__()
        self.channelblock=nn.Sequential(
            nn.Linear(CHANNELS,DC),
            nn.BatchNorm1d(PATCHES),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(DC,CHANNELS),
            )
        
    def forward(self,x):
        return self.channelblock(x)
class MixerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_mixing=TokenMixer()
        self.channel_mixing=ChannelMixer()
        
    def forward(self,x):#patches*channels
        global CHANNELS
        y=nn.LayerNorm(CHANNELS)(x)
        y=torch.permute(y,(0,2,1))
        y=self.token_mixing(y)
        y=torch.permute(y,(0,2,1))
        z=x+y
        zz=nn.LayerNorm(CHANNELS)(z)
        zz=self.channel_mixing(zz)
        return zz+z
from typing import OrderedDict
class Mixer(nn.Module):
    def __init__(self,num):
        global CHANNELS,PATCHES,P
        super().__init__()
        self.patch_connect=nn.Conv2d(3, CHANNELS, P,stride=P)
        blocks=OrderedDict()
        for i in range(num):
            blocks['MixerLayer '+str(i)]=MixerLayer()
        self.mixerblocks=nn.Sequential(blocks)
        self.full_connect=nn.Linear(PATCHES,100)
        
    def forward(self,x):
        global CHANNELS,PATCHES
        # y=nn.BatchNorm2d(3)(x)
        y=self.patch_connect(x)
        y=nn.Flatten(start_dim=2)(y)#channels*patches
        y=torch.permute(y, (0,2,1))
        y=self.mixerblocks(y)
        y=nn.LayerNorm(CHANNELS)(y)
        #全局最大池化
        y=nn.MaxPool1d(CHANNELS)(y)
        y=nn.Flatten()(y)
        y=self.full_connect(y)
        return y

mixer=Mixer(12)
#输出模型参数总量
from pytorch_model_summary import summary
print(summary(mixer,torch.randn(1,3,SIZE,SIZE), show_input=False, show_hierarchical=False))


#数据预加载
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose,ToTensor,Lambda,Normalize,Resize
class Mydata(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.transform=Compose(
            [
                ToTensor()
            ]
            )
        self.target_transform=Lambda(lambd=lambda y: torch.zeros(100, dtype=torch.float).scatter_(0, y.clone().detach(), value=1))
        self.y=y
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        data=self.transform(self.X[idx])
        label=self.target_transform(torch.tensor(self.y[idx]))
        return data.to('cuda'),label.to('cuda')

traindata=Mydata(Xtrain,ytrain)
testdata=Mydata(Xtest,ytest)

trainloader=DataLoader(traindata,batch_size=1024,shuffle=True,generator=torch.Generator(device='cuda'))
testloader=DataLoader(testdata,batch_size=1024,shuffle=False,generator=torch.Generator(device='cuda'))

from torch.nn import CrossEntropyLoss,init
from torch.optim import Adam,SGD,lr_scheduler
from torch.nn.utils import clip_grad_norm_

mixer=Mixer(12).to('cuda')
# 自定义初始化
for name,params in mixer.named_parameters():
    init.normal_(params)

with torch.device('cuda'):
    loss_fn=CrossEntropyLoss()#Softmax included
    #warmup
    optimizer=Adam(mixer.parameters(),lr=0.1)
    epoches=100000
    # scheduler=lr_scheduler.LinearLR(optimizer,start_factor=1e-5,end_factor=1,total_iters=epoches*len(trainloader))
    scheduler=lr_scheduler.LinearLR(optimizer,start_factor=1,end_factor=1,total_iters=epoches*len(trainloader))
    losslist=[]
    for i in range(epoches):
        print(f'\nepoch {i+1} start')
        avgloss=0
        mixer.train()
        for idx,(X,y) in enumerate(trainloader):
            pred=mixer(X)
            loss=loss_fn(pred,y)
            loss.backward()
            biasmaxlist=[]
            weightmaxlist=[]
            for name,params in mixer.named_parameters():
                if 'bias' in name:
                    biasmaxlist.append(params.grad.max().item())
                else:
                    weightmaxlist.append(params.grad.max().item())
            clip_grad_norm_(mixer.parameters(), max_norm=1,norm_type=1)#gradient-clipping  if ratio>1,then scale return total_norm
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            avgloss+=loss.item()
            print(end='>')
            plt.plot(weightmaxlist)
        plt.show()
        avgloss=avgloss/(idx+1)
        print(f'\ntrain loss:{avgloss}')
        losslist.append(avgloss)
        
# with torch.device('cuda'):
        mixer.eval()
        acc=0
        acclist1=[]
        avgloss1=0
        loss_fn=CrossEntropyLoss()#Softmax included
        # y_pred=[]
        # y_true=ytest
        with torch.no_grad():   
            for idx,(X,y) in enumerate(testloader):
                pred=mixer(X)
                avgloss1+=loss_fn(pred,y).item()
                acc+=sum(pred.argmax(dim=1)==y.argmax(dim=1)).item()
                # y_pred+=pred.argmax(dim=1).tolist()
                print(end='>')
            acc/=len(testloader.dataset)
            acclist1.append(acc)
            print(f'\ntest accuracy:{acc}')
    
torch.save(mixer.state_dict(),'../MixerReplication/mixer.params') 


