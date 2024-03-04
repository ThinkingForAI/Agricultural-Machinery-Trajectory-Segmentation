import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
#加载当前目录下的所有文件 返回X,y
import os
path='Features25/wheat1'
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
        temp=pd.read_excel(os.path.join('Features25/wheat1',tablename))
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
    print(f'The {i}th tracjectory has processed!')
    j+=1        
    return X,y #numpy:Numbers*Features*Channels  list:label
#采用划分后的数据
X_train,y_train=sliding_channel_windows(train)
X_valid,y_valid=sliding_channel_windows(valid)
X_test,y_test=sliding_channel_windows(test)

#CV公用数据集的使用
from torchvision import datasets
from torchvision.io import read_image
# 自定义数据集
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import Lambda

class mydataset(Dataset):
    #data label均是列表形式
    def __init__(self,data,label,mode='valid'):
        #转换成张量
        #(n,25,9)->(n,25,3,3)
        data=data.reshape(len(data),25,3,3)
        #通道增加
        self.data=torch.tensor(data)
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


from torch import nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(25,64, 1),#c1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128,1),#c1
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*3*3, 2),
            nn.Softmax(dim=1)
            )
    def forward(self,x):
        prob=self.layers(x)
        return prob

def train_eval(DataLoader,model,loss_fun,optimizer):
    size=len(DataLoader)#每个batch的大小为64，size为batch的总量，对valid一共1773个batch
    
    model.train()#说明：dropout和BN在训练和验证时的作用不同
    
    # list(enumerate(validDataLoader))[0][1][0].shape
    # Out: torch.Size([64, 1, 10, 10])\
    #一个batch的训练是同时进行的
    
    avg_loss=0
    for batch,(X,y) in enumerate(DataLoader):#仅仅只是增加了序号
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
#--------------------------训练--------------------------------------#


def read():
# #如何从DataLoader读出数据？
# #next+iter方法
# data,labels=next(iter(validDataLoader))#一次返回一个batch 结构A*B*C

# #收缩维度 tensor.squeeze()
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 2, 2
# import random
# labels_map=['road','field']
# for i in range(1, cols * rows + 1):
#     idx=random.randint(0,len(data)-1)
#     img=data[idx].squeeze()
#     label=labels[idx].argmax(0)#在第0维上的最大值
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.to('cpu'), cmap="gray")
# plt.show() 
    return


mlp=MLP().to('cuda')#模型的整个执行过程在GPU上进行 
# print(mlp)
#输出模型参数总量
from pytorch_model_summary import summary
tmp = torch.randn(1, 25, 3, 3).to('cuda')
print(summary(mlp,tmp, show_input=False, show_hierarchical=False))

# #遍历模型初始化的所有参数
# for name,params in mlp.named_parameters():
#     print(f'Current Layer:{name},Parameters{params}')
#     break

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


    
    
    
    
    
    

    