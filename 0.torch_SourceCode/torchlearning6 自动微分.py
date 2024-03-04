# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:18:54 2024
@author: DELL
"""
#-------------AutomaticDifferentiation(AD)-----------------------#
# torch.Tensor.detach 脱离计算图但共享内存
# torch.Tensor.clone 仍在计算图上，但不共享内存，视为一个一一映射函数 
# torch.Tensor.copy_ 数据广播式覆盖 保留原有设定

#前提：标量值函数
#-------------AutomaticDifferentiation(AD)-----------------------#
import numpy as np
import torch
w=torch.tensor(1.,requires_grad=True)
x=torch.tensor(2.,requires_grad=True)
# y=(x+w)*(w+1)
a=torch.add(w,x)
'''
a.retain_grad()
'''
b=torch.add(w,1.)
y=torch.mul(a,b)
#每个非叶节点保存了计算该非叶节点的函数
y.grad_fn


torch.autograd
#计算偏导数
y.backward()
#print(a.grad) #报错
print(w.grad) #正确 只有计算图上的叶结点梯度才能被保存
#是否是叶子结点
print(w.is_leaf)

#注：在计算图计算的过程中，非叶结点的梯度是存在的，但是计算一旦完成，非叶结点的梯度即被释放，目的是为了节省空间

#如何保存非叶结点的梯度？
a.retain_grad()

#Tensorflow采用静态图 预先搭建  
#pytorch采用静态图 运算和搭建同时进行


#------------------------------前向计算--------------------------------#
#前向自动微分计算forwardAD
'''
偏导数 partial derivatives
方向导数 directional derivatives
'''
#向量值函数的梯度
x=torch.tensor([1.,2.,3.],requires_grad=True)
y=x**2
y.backward(torch.tensor([2,2,2]))
#vT·J vT默认为{1,.....1}
x.grad

#术语:输入Primal 方向Tangent dual-tensor指不仅含有该向量的值还含有前向梯度
import torch
import torch.autograd.forward_ad as fwAD
primal=torch.tensor([[1.,2.],[3.,4.]])
tangent=torch.tensor([[1.,1.],[2.,2.]])
#tangent指明方向导数的方向
def fn(x,y):
    return x**2+y**2
with fwAD.dual_level():
    dual_input=fwAD.make_dual(primal, tangent)
    assert fwAD.unpack_dual(dual_input).tangent is tangent
    # dual_input_alt = fwAD.make_dual(primal, tangent.T)
    # assert fwAD.unpack_dual(dual_input_alt).tangent is not tangent
    plain_tensor = torch.tensor([[1.,1.],[1.,1.]])
    dual_output = fn(dual_input, plain_tensor)
    # Unpacking the dual returns a ``namedtuple`` with ``primal`` and ``tangent``
    # as attributes
    #返回计算的值 和 雅可比向量积
    y,jvp = fwAD.unpack_dual(dual_output)#jvp=J(element-wise multi)tangent
assert fwAD.unpack_dual(dual_output).tangent is None

#前向计算的使用
import torch.nn as nn
model = nn.Linear(5, 5)
input = torch.randn(16, 5)
params = {name: p for name, p in model.named_parameters()}
tangents = {name: torch.rand_like(p) for name, p in params.items()}
with fwAD.dual_level():
    for name, p in params.items():
        delattr(model, name)
        setattr(model, name, fwAD.make_dual(p, tangents[name]))#生成dual_tensor
    out = model(input)
    jvp = fwAD.unpack_dual(out).tangent


#前向计算的使用
import torch.nn as nn
model = nn.Linear(2, 2)
inputs = torch.tensor([[1.,1.],[2.,2.]])
y=torch.sum(model(inputs))
y.backward()


#------------------------------Autograd进阶-------------------------------#
#高级函数
from torch.autograd import functional
def fn(inputs,*args,**kwargs):
    return inputs**2

inputs=torch.tensor([1.,2.])
jacobian_matrix=functional.jacobian(lambda x:fn(x), inputs)

#输出单个元素才用hessian矩阵
def fn1(inputs,*args,**kwargs):
    return sum(inputs)
hessian_matrix=functional.hessian(fn1, inputs)

y1,jvp=functional.jvp(fn, inputs,v=torch.tensor([1,1]))

y2,hvp=functional.hvp(fn1, inputs,v=torch.tensor([1,1]))

a=torch.tensor(2.,requires_grad=True)
y=a+2
y.backward()
#梯度不清空则梯度累计 a.grad=None
y=a**2
# a.grad=None
# y.backward(create_graph=True)#前提上一次使用的a.grad要重新设置为None否则会造成内存泄露 尽量不用

#梯度清零
# model.zero_grad()    optimizer.zero_grad()


#---------------自定义forward和backward方法（针对不可导函数！）---------------#
import torch
from torch.autograd import Function
class myfun(Function):
    @staticmethod 
    def forward(ctx, inputs1,inputs2):
        result=sum(inputs1**2)+sum(inputs2**2)
        #后向传播所使用的变量
        interm=inputs1
        ctx.save_for_backward(interm)
        return result 
    
    @staticmethod 
    def backward(ctx, grad_outputs):#链式法则 给定输出的梯度计算输入的梯度
        #grad_outputs默认值为1
        print(grad_outputs)
        #梯度计算
        interm,=ctx.saved_tensors
        print(interm)
        '''
        ctx.saved_tensors返回元组(tensor([2., 3.], requires_grad=True),)但实际需要的是元素
        '''
        print(ctx.needs_input_grad) #(True, False)
        grad_in=2*interm
        ##ctx.needs_input_grad作为布尔张量，用来反映每一个input是否需要计算梯度（元组，不允许修改值
        return grad_in*grad_outputs,None #对应inputs的梯度
x1=torch.tensor([2.,3.],requires_grad=True)
x2=torch.tensor([2.,3.],requires_grad=False)
a=myfun.apply(x1,x2)#通过apply方法调用forward方法
a.backward()

#---------------计算图的观测---------------#
from torch.autograd.graph import Node
x1=torch.tensor([2.,3.],requires_grad=True)
x2=torch.tensor([1.,1.],requires_grad=True)
y1=sum(x1)
y2=sum(x2)
y=y1+y2
#计算图上的结点是由.grad_fn属性维持的,为了后向传播使用
assert isinstance(y.grad_fn, Node),'Not the same type'

#当前结点名称
y1.grad_fn.name()
#当前结点下一结点（属性）
y1.grad_fn.next_functions
y1.grad_fn.metadata
'''
@abc.abstractmethod
def _register_hook_dict(self, tensor: torch.Tensor) -> None:

@abc.abstractmethod
def register_hook(self, fn: Callable[..., Any]) -> RemovableHandle:
'''


#------------------------------hook技术----------------------------------#
'''
1.torch.Tensor.register_hook 在backward()后被调用
signature::
    hook(grad) -> Tensor or None
'''
#hook函数输出模型中间信息
import torch
a=torch.tensor(2.,requires_grad=True)
aa=3*a
y=a**2
# y=aa**2
#如何输出aa的梯度?
tempgrad=None
def hook_fn(grad):#该hook函数仅处理梯度 
    global tempgrad#结果返回给全局变量  变量局部性质 列表全局性质
    tempgrad=grad
    return torch.tensor(1.)#通过返回值修改原值
hookhandle=a.register_hook(hook_fn)
y.backward(retain_graph=True)#保证多次反向传播 但是梯度累计

#移除hook hookhandle.remove()

import torch
from torch.nn import Module,Sequential,Linear
class net(Module):
    def __init__(self):
        super().__init__()
        self.layers1=Sequential(
            Linear(2, 2),
            Linear(2,3)
            )
        self.layers2=Sequential(
            Linear(3, 2),
            Linear(2,1)
            )
    def forward(self,x):
        l1=self.layers1(x)
        l2=self.layers2(l1)
        return l2
    
    
'''
2.torch.nn.Module.register_forward_hook #在forward后被调用
'''
m=None
def forwardhook_fn(module,inputs,outputs):#可以更改输入输出 但只有输出对后向传播造成影响
    #ps:导出的参数转存cpu，不要放在GPU上
    global m
    m=module
    # outputs.add_(1) #修改输出
    
Net=net()
hookhandle2=Net.register_forward_hook(forwardhook_fn,)

y=Net(torch.tensor([1.,2.]))
#hookhandle2.remove()

'''
3.torch.nn.Module.register_forward_pre_hook #在前向计算前修改输入

'''
def forwardhookpre_fn(module,inputs):
    print(module)
    # inputs[0].add_(1)#在前向计算前修改输入

   
Net=net()
hookhandle3=Net.register_forward_pre_hook(forwardhookpre_fn,)

y=Net(torch.tensor([1.,2.]))
#hookhandle3.remove()

'''
4.torch.nn.Module.register_backward_hook #即将被弃用
新版本
torch.nn.Module.register_full_backward_hook 参数 Module grad_in grad_out
#返回修改后的输入梯度的新值参与后续计算

register_full_backward_pre_hook 参数 Module grad_out
#返回修改后的输出梯度的新值参与后续计算

#均不能修改原值
'''
def backwardhook_fn(module,grad_in,grad_out):
    # grad_input=grad_in
    print(Module)
    print(grad_in)
    print(grad_out)#禁止原地改动
    # return grad_input

Net=net()
'''
!!!!!注册到层上
对于layers2来说 输入 输出均是变量
但对于整个模型Net来说 输入是常量
# Net.register_full_backward_hook(backwardhook_fn)
'''
hookhandle4=Net.layers2.register_full_backward_hook(backwardhook_fn)
y=Net(torch.tensor([1.,2.]))
y.backward()
#hookhandle4.remove()
#综上所述，hook函数的定义紧随实例化的模型之后














