# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:52:41 2024

@author: DELL
"""

#------------------torch.nn-------------------------------#
'''网络架构每层所用函数的抽象实现、各层类的定义、损失函数类'''


#-----------------模型参数类Parameter-------------#
import torch
from torch.nn.parameter import Parameter
a=torch.tensor([1.,2.])
x=Parameter(a)#自动允许计算梯度
'''
x
Out[192]: 
Parameter containing:
Parameter([1., 2.], requires_grad=True)
'''
# 如何修改值？如何取值？
#取值
y1=x.clone().detach()

#改值
y2=x.detach()
y2.add_(1)
'''
y2
Out[201]: tensor([2., 3.])

x
Out[202]: 
Parameter containing:
Parameter([2., 3.], requires_grad=True)
'''

#-----------------Sequential类-------------#
import torch
from torch import nn
from typing import OrderedDict
layers=torch.nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1,20,5)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(20,64,5)),
        ('relu2', nn.ReLU())
    ]))

layers2=torch.nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
    )


#-----------------ModuleList类-------------#
from torch import nn
linears=nn.ModuleList([nn.Linear(10,10) for i in range(10)])
linears[1]
#-----------------Moduledict类-------------#
from torch import nn
activations=nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
    ])
activations['lrelu']


#------------------卷积类----------------#
#序列数据1D卷积
'''
#实际卷积核大小 batch*2D
'''
from torch import nn
conv1d=nn.Conv1d(in_channels=4,out_channels=1,kernel_size=2)
x=torch.tensor([
    [1.,1.,1.,1.,1.],
    [2.,2.,2.,2.,2.],
    [3.,3.,3.,3.,3.],
    [4.,4.,4.,4.,4.]
    ])
'''
#从时间序列数据上理解：
有L条时间序列数据，每条数据有K个特征，则通道数为K，每个通道L个数参与卷积
例子：
输入Cin=4 输出Cout=1 卷积核规定窗口大小为2，实际是Cin*2=4*2 偏置Cout=1
所以一维卷积对特征全部卷积
'''

#图像数据2D卷积 
'''
#实际卷积核大小 batch*3D
'''
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np
im=Image.open('../CNN/exam.jpg')
im_tensor=transforms.ToTensor()(im)
conv2d=nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(2,2))
transforms.ToPILImage()(conv2d(im_tensor))


#视频处理3D卷积
'''
#实际卷积核大小 batch*4D

4D:3*t(时序长度)*H*W
通过设置卷积核的大小,可是实现时间轴上的滑动窗口
'''

#视频读入
import torchvision
#视频公用数据集UCF101
from torchvision.datasets import UCF101

'''
上采样(Unsampling)的方法
双线性插值(bilinear)，反卷积(Transposed Convolution)，反池化(Unpooling)

'''
#反卷积Deconvolution
nn.ConvTranspose1d


#------------------池化类-------------------------------#



#------------------padding类----------------------------#


