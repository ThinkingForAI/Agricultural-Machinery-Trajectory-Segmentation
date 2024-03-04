# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 00:20:52 2024

@author: DELL
"""
import torch


#迭代器的两个基本方法
# iter() next()
a=[1,2,3,4]
b=iter(a)
while 1:
    try:
        next(b)
    except StopIteration:
        break
    

#创建迭代器
class myiterator:
    def __iter__(self):
        self.a=1
        return self #return self的目的是为了链式调用 aa.__iter__().__iter__()(类的实例)
    def __next__(self):
        x=self.a
        self.a+=1
        if self.a >=3:
            raise StopIteration
        return x

#迭代器定义是并没有占用空间，仅仅只有在iter(aa)时候才实例化
aa=myiterator()
bb=iter(aa)



#生成器
#以斐波那契数列的改进为例
#version 1
def fib(lim):
    n,a,b=0,0,1
    L=[]
    while n<lim:
        L.append(b)
        a,b=b,a+b
        n+=1
    return L
#L的内存占用越来越大

#version 2
class Fib():
    def __init__(self,lim):
        self.lim=lim
        self.n,self.a,self.b=0,0,1
    def __iter__(self):
        return self
    def next(self):
        if self.n<self.lim:
            self.a,self.b=self.b,self.a+self.b
            self.n+=1
            return self.b
        else:
            raise StopIteration

#内存占用为常数

#version 3
def fib(lim):
    n,a,b=0,0,1
    while n<lim:
        yield b
        a,b=b,a+b
        n+=1
#简洁版本
#yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator
#每次迭代的值就是yield后变量的值


#结构化python编程
'''
例子
-temp
-----__init__.py  print('init')
-----a.py print('a')
#------out------#
import temp #重复运行只有第一次有效  等价与 import temp.__init__
#[out]:init
'''
#每一个文件夹是一个package 每一个py文件是一个module    import文件夹则文件夹下仅有__init__.py被调入
'''
1.当导入一个包时，Python会自动执行该包下的__init__.py文件。这意味着在导入包的时候，__init__.py文件中的代码会被执行。

2.当使用import语句导入一个模块时，如果该模块所在的目录中存在__init__.py文件，那么该__init__.py文件也会被执行。
例1：
import temp.a
init
a
例2：
import temp.__init__
init
init
'''
#调用特定模块
"""
1.from 包 import 模块
2.from 包 import *(默认只调入__init__),还需在__init__.py中定义__all__属性=['模块名']
"""
# __all__属性,在__init__.py文件中定义可以被外界调用的类和方法



#import 加载包 模块


#torch.utils.data.sampler类详解
'''
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)
'''
#torch.randperm(100) 产生随机排序序列





    