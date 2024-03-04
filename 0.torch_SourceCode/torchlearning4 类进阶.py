# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:32:19 2024

@author: DELL
"""
#----------方法、属性----------#
class example:
   #静态属性 不需声明
   discount=0.5
   def __init__(self,name,age):
       self.name=name
       self.age=age
   
   #静态方法需声明
   @staticmethod #静态方法不能访问类的实例属性 也用于定义类的静态属性
   def staticM(value):
       example.discount=value
       example.name='apple'
      
   #类方法
   @classmethod #常用于处理类实例相关
   #第一个参数为cls,代表调用该方法的类
   def from_string(cls, sentence):
       name = sentence#从句子中解析返回实例化结果
       age = sentence
       return cls(name, age)

#----------继承----------#
class a :   
    def __init__(self,value):
        print('a constructor function is called')
        self.value=value
        self.vv1=value
    
    def name(self):
        print(f'a {self.value}')
    
    @staticmethod #不需要实例化就能使用 类专用的工具函数
    def a_static_method():
        print('a static method')
    
    #双置前下划线定义类的私有属性
    def __name2(self):
        print(f'a {self.value+1}')
    def name3(self):
        print(f'a {self.vv1+1}')
    
    def __del__(self):
        print('a destructor function is called')


class b:
    def __init__(self,value):
        self.value=value
        self.vv2=value
        print('b constructor function is called')
        
    def name(self):
        print(f'b {self.value}')
      
    def name2(self):
        print(f'b {self.value+1}')
        
    def __del__(self):
        print('b destructor function is called')
        

class aa(a,b):
    def __init__(self,value):
        print('aa constructor function is called')
        self.value=value
    # #子类调用父类方法
    # def parent_name(self):
    #     a.__init__(self,self.value)
    #     a.name(self)
    
    
    #super()存在的意义：调用被重写的父类方法 菱形继承
    #super(type1,obj or type2) #type1表明类type2的__mro__属性的起始位置 通过类的实例obj或者类type2指明是谁的__mro__属性
    #-------------------当参数是obj时返回类的实例 当参数是type时返回类的代理 
    #                   使用super(D,F()).fun() super(D,F).fun(F())
    #super()自动填充当前类 当前类实例 (用在类体内部)
    
    def parent_name(self):
        super().__init__(self.value)
        super().name()
    
    
    
    def name(self):
        print(f'aa {self.value}')
        
    def __del__(self):
        print('aa destructor function is called')
        

#调用mro方法查看继承的所有顺序  Method Resolution Order


#----------有向图 先左后右 消入度为0的边----------#
class A(object):
    def fun(self):
        print('A.fun')

class B(object):
    def fun(self):
        print('B.fun')

class C(object):
    def fun(self):
        print('C.fun')

class D(A,B):
    def fun(self):
        print('D.fun')

class E(B, C):
    def fun(self):
        print('E.fun')

class F(D, E):
    def fun(self):
        print('F.fun')

        
