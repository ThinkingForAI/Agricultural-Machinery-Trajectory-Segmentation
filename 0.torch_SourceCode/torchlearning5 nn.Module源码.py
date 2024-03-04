# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:57:52 2024

@author: DELL
"""
'''
#所有类默认继承自object类  object类自带属性如下
['__class__',
 '__delattr__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__getstate__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__']
'''
class example(object):
    def __init__(self):
        print('into init')
        self.para=1#调用__setattr__方法
    #子类的全局变量和函数放在子类的dict中，父类的放在父类dict中。
    def __setattr__(self,keystr,value):#'value'传给keystr 1传给value
        print('into setattr')
        # self.keystr=value #死循环 循环调用__setattr__
        super().__setattr__(keystr, value) #等价于 self.__dict__[keystr]=value
    
    def __getattr__(self,keystr):
        print('into getattr')
        #a=example() 运行a.para即调用a.__getattr__('para')
        if keystr in self.__dict__.keys():
            print(self.__dict__[keystr])
        else:
            print(f'{self.__class__} object has no attribute {keystr}')


from typing import * #导入更多类型 Tuple List Dict Set Union Any Optional Callable

class exam:
    #类型声明仅起到提示和注解的作用  不会被检查错误
    attr1:str
    attr2:int
    #类型声明 有具体的值表示为静态属性
    can_superinit:bool=True
    
    def __init__(self,*args,**kwargs):
        if self.can_superinit:
            super().__init__(*args,**kwargs)
        # else:
        #     super().__setattr__('attr1', 'exam1')
        #     super().__setattr__('attr2', 1.2)
        else:
            self.attr1='exam1'
            self.attr2=1.2
    
    def __setattr__(self,keystr,value):
        print('into subclasss setattr')
        super().__setattr__(keystr, value)\
   
    
# ---------------------装饰器---------------------#
# python函数修饰符@的作用是为现有函数增加额外的功能，常用于插入日志、性能测试、事务处理等等。
# 创建函数修饰符的规则：
# （1）修饰符是一个函数
# （2）修饰符取被修饰函数为参数
# （3）修饰符返回一个新函数
# （4）修饰符维护被维护函数的签名  
#修饰函数 @函数A  其后紧跟的函数B为函数A的参数
def funA(funB):
    print('funA is called,param is funB')
    a,b=1,2
    return funB(1,2)

@funA  #注释：dir(funA) 有__call___
def funB(x,y):
    print('funB is called,to calculated the add')
    return x+y
# 在PyTorch中，张量的 requires_grad 属性的默认值是 False。这意味着创建的张量默认情况下不会被跟踪梯度，也不会参与反向传播过程
import torch
from torch import nn
@torch.no_grad() #注释：no_grad类实例化，继承的父类有__call__方法 @torch.no_grad()是通过类的实例调用 而@torch.no_grad是通过类名直接调用
#确保张量在函数中计算的结果都不计算梯度
def x():
    a=torch.tensor(1).float()
    a.requires_grad=True
    #要计算梯度的变量是不能修改的
    '''
    a.add_(2)
    RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
    '''
    return a.add(1)
a=x()

class A(object):
    def __init__(self, value):
        self.value = value
 
    def __getattr__(self,attributename):#调用不存在的方法时会自动调用__getattr__方法,参数为方法名
        print ("into __getattr__")
        print(attributename)
 
a = A(10)
print(a.value)
a.name

import torch      
#-----------------------------------模型训练----------------------------------#
#代码详细解读
class mynet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1=torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            torch.nn.ReLU(),       torch.nn.Linear(3, 1),
            # torch.nn.Softmax(dim=1)
            )        
    def forward(self,x): #等价于抽象的__call__方法1
        return self.layers1(x)
    
    '''
    第一次net(x)调用mynet继承的nn.Module的__call__方法1,也就是调用自身的forward方法;
    第二次self.layers(x)调用nn.Sequential继承的nn.Module的__call__方法2,
    也就是调用nn.Sequential的forward方法,最终结果为该方法的输出
    ps:访问类内不存在的属性就会触发__getattr__方法（前提是实现该方法）,由于net类没有__getattr__方法,所以没触发
    '''
#有亿点点小小的问题关于layers call调用时先调用getattr？？？？？？？
net=mynet()
x=torch.tensor([1,2]).float()
a=net(x)

     

