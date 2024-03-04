# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:44:38 2024

@author: DELL
"""
#Miller Projection
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#定义经纬度转换为米勒坐标的方法
def millerToXY(lon, lat):
    L = 6381372 * math.pi * 2  #地球周长
    W = L  #平面展开，将周长视为X轴
    H = L / 2  #Y轴约等于周长一半
    mill = 2.3  #米勒投影中的一个常数，范围大约在正负2.3之间
    #循环，因为要批量转换
    xlist=[]
    ylist=[]
    for x, y in zip(lon, lat):
        x = x * math.pi / 180  # 将经度从度数转换为弧度
        y = y * math.pi / 180  # 将纬度从度数转换为弧度
        y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))  # #这里是米勒投影的转换
        x = (W / 2) + (W / (2 * math.pi)) * x  #这里将弧度转为实际距离 ，转换结果的单位是km
        y = (H / 2) - (H / (2 * mill)) * y  # 这里将弧度转为实际距离 ，转换结果的单位是km
        xlist.append(x)
        ylist.append(y)
    return xlist,ylist
#坐标平移
def translation(X,Y):
    Xnew,Ynew=np.array(X),np.array(Y)
    Xnew,Ynew=Xnew-min(Xnew),Ynew-min(Ynew)
    return Xnew,Ynew

#输出实际轨迹
def draw_trajectory_picture(data):
    #按标签分别转换
    data_f,data_r=data[data['tags']==1],data[data['tags']==0]
    x_f,y_f=millerToXY(data_f['lon'], data_f['lat'])
    x_r,y_r=millerToXY(data_r['lon'], data_r['lat'])
    if len(x_f):#考虑标签单一的情况
        x_fnew,y_fnew=translation(x_f, y_f)
        plt.plot(x_fnew,y_fnew,'g')   
    if len(x_r):
        x_rnew,y_rnew=translation(x_r, y_r)
        plt.plot(x_rnew,y_rnew,'b') 
    # plt.figure(facecolor='black')
    # plt.axis('off')
    plt.show()

if __name__=='__main__':
    
    os.path.join('')




