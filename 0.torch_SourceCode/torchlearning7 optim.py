# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:52:41 2024

@author: DELL
"""
#----------------优化方法----------------#
'''
torch.optim 文件夹下有各种优化方法（sgd...）的py文件，以及optimizer.py文件，各种优化方法例如sgd.py中的SGD继承自optimizer.py中的Optimizer类

'''
import torch
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

'''
模型定义
'''
from torch import nn
class examnet(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.layer1=nn.Linear(2,2)
        self.layer2=nn.Linear(2,1)
    def forward(self,x):
        return self.layer1(x)

xx=torch.tensor([1.,2.])
model=examnet()

#
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
'''
dir查看类的所有属性和方法 vars查看类的实例的属性 hasattr查看属性是否存在 getattr获取属性的值
'''
#查看优化器中的模型所有参数（包括超参数）
optimizer.param_groups[0]#获取字典
optimizer.param_groups[0]['params']#获取模型参数  通过.grad方法可以访问参数的梯度
#ps:与网络中的参数共享内存


#----------------个性化设置模型参数,按照输出格式----------------#
'''
{'params': [Parameter containing:
  tensor([[ 0.6084, -0.5190],
          [ 0.5252, -0.2944]], requires_grad=True),
  Parameter containing:
  tensor([0.0659, 0.3713], requires_grad=True),
  Parameter containing:
  tensor([[ 0.1950, -0.1819]], requires_grad=True),
  Parameter containing:
  tensor([-0.1768], requires_grad=True)],
 'lr': 0.01,
 'momentum': 0.9,
 'dampening': 0,
 'weight_decay': 0,
 'nesterov': False,
 'maximize': False,
 'foreach': None,
 'differentiable': False}
'''
w1=torch.tensor([[1,0],[0,1]])
b1=torch.tensor([1,1])
#每层传递一个参数字典,所有层形成列表
optimizer=optim.SGD(
    [
     {'params':[w1,b1],'lr':10,'momentum':5},
     {'params':model.layer2.parameters(),'lr':0.1,'momentum':2.5}
    ],lr=0.01,momentum=0.05)#如果层内没规定lr则用外面的


#梯度置空方法
optimizer.zero_grad()

#梯度更新
optimizer.step()


#----------------学习率调度器torch.optim.lr_scheduler----------------#
from torch.optim import lr_scheduler

'''
#动态学习率调整
lr_scheduler.ReduceLROnPlateau()
'''

#如何使用？
#例：指数衰减学习率ExponentialLR(optimizer, gamma=0.9)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#在每个epoch迭代后使用scheduler.step() 在optimier.step()之前使用会调过学习率更新的第一个值
scheduler.get_lr() #获取当前学习率
scheduler.step()
scheduler.get_last_lr()#获取上一次迭代的学习率





