# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 22:31:50 2024

@author: DELL
"""

import numpy as np
import torch 
data=[[1,2],[3,4]]
x_data=torch.tensor(data)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
x_zeros=torch.zeros_like(x_data)

# 1.1  定义
#1 定义 CPU、GPU存储位置移动
tensor=torch.rand(3,4)
if torch.cuda.is_available():
    tensor=tensor.to('cuda')
print(tensor.device)
t1=torch.cat([tensor,tensor],dim=1)


#2 取数
tensor[1,2].item()

#3 原地操作
tensor.add_(1)

#4 cpu上的tensor与numpy互转
tensor=tensor.to('cpu')
#共享空间
t=tensor
n=tensor.numpy()
u=torch.from_numpy(n)


# 1.2  Datasets & Dataloaders
#公用数据集 torchvision.datasets
import torchvision
#1 FashionMNIST
#默认加载的时训练集
data=torchvision.datasets.FashionMNIST('../MNIST',download=True)
data_loader=torch.utils.data.DataLoader(data,batch_size=5,shuffle=True,num_workers=0)
#torch.utils.data.DataLoader 多进程并行执行 num_workers进程数量（设置为CPU核心数），batch_sampler将指定batch分配给指定worker，worker将它负责的batch加载进RAM(加快训练速度)

# torch.utils.data.Dataset() 存储(X,y)
# torch.utils.data.DataLoader() 打包成itreable容器便于加载

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

train_data=datasets.FashionMNIST('../MNIST',train=True,download=False,transform=ToTensor())
test_data=datasets.FashionMNIST('../MNIST',train=False,transform=ToTensor())

# test_data[0][0].shape
# Out: torch.Size([1, 28, 28])
# test_data[0][1]
# Out: 9
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    #函数返回两个参数X,y
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    # tensor.squeeze() (A×1×B×C×1×D)->(A×B×C×D)
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


#2 Creating a Custom Dataset for your files
import os
import pandas as pd
from torchvision.io import read_image
#read_image读入图片后自动按照三通道划分

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)#读入的图片就是torch
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



#3 Preparing your data for training with DataLoaders
from torch.utils.data import DataLoader
train_dataloader=DataLoader(train_data,shuffle=True,batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
#Each iteration below returns a batch,Display image and label.迭代完则打乱
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")


# 1.3  Transforms
#1 All TorchVision datasets have two parameters -transform to modify the features and target_transform to modify the labels
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="../MNIST",
    train=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
#lambda one-hot encode
#ToTensor()转成Tensor并且标准化[0,1]
#Tensor.scatter(dim, index（tensor）, src) 

# 1.4 BuildModel
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

device=('cuda' if torch.cuda.is_available() else 'mps' 
        if torch.backends.mps.is_available() else 'cpu')


#1 define the class
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            )
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

model=MLP().to(device)
print(model)

# Do not call model.forward() directly!
X=torch.rand(1,28,28,device=device)
logits=model(X)
pred_probab=nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
        
#2 detail
input_image=torch.rand(3,28,28)#默认从第一维开始 实际输入第0维为batch 最外层为第0维
flatten=nn.Flatten(start_dim=1,end_dim=-1)
flat_image=flatten(input_image)

layer1=nn.Linear(28*28, 20) 
hidden1=layer1(flat_image)  
hidden1=nn.ReLU()(hidden1)

#建模
seq=nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10),
    nn.Softmax(dim=1),#Softmax指明进行运算的维度
)     

#3 模型参数
#模型结构 model 
for name,param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        
        
        
    
# 1.6 Optimization
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="../MNIST",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="../MNIST",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
#加载模型参数
# model.load_state_dict(torch.load('model_weights.pth'))
lr=0.5
batch_size=64
epoches=5

#后向传播
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#optimizer[0]为字典存放参数 optimizer.param_groups[0]['params']存放的是需要更新的模型参数

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


#保存模型参数
torch.save(model.state_dict(), 'model_weights.pth')

#保存模型
torch.save(model, 'model.pth')
#加载模型
model = torch.load('model.pth')