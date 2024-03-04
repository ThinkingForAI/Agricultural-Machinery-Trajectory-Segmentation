# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 23:05:35 2024

@author: DELL
"""
import pandas as pd
import datetime
import os
from math import sin,cos,asin,sqrt,radians
import numpy as np
import visualize

#标准最大速度
DEFINEMAXSPEED=60

#实际距离（m)
def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

#数据预处理 去异常 去停 去空白
def data_pre_process(data,
                     anomalousPointEnable=True,
                     stopPointEnable=False,
                     whitePointEnable=True):
    """
    data:未处理前的经纬度等原始数据
    anomalousPointEnable:启用处理异常点
    stopPointEnable:启用处理停止点
    whitePointEnable:启用处理空白点
    """
    #查重duplicated 去重drop_duplicates
    #去除时间重复点
    data.drop_duplicates(subset=['time'],keep='first',ignore_index=True,inplace=True)
    
    if whitePointEnable:#空白点
        data.dropna(subset=['lat','lon'],inplace=True,how='any',axis=0)
        #索引重排
        data.reset_index(drop=True,inplace=True)
        
    if stopPointEnable:#经度 维度 速度 方向相同(静止点)
        data.drop_duplicates(subset=['lon','lat','speed','dir'],keep='first',ignore_index=True,inplace=True)
        
    if anomalousPointEnable:#异常点
        deletedIndex = []

        lonlist = data['lon'].tolist()
        latlist = data['lat'].tolist()
        speedlist = data['speed'].tolist()
        
        speed_iszero = (np.array(speedlist) == 0).all()  ## 如果所有的速度都是0，则按照经纬度计算距离平均速度更新速度
        
        maxspeed = DEFINEMAXSPEED * 0.2777778#最大速度上限为60/3.6=16.67m/s
        
        try:
            timelist=[datetime.datetime.strptime(str(time),'%Y/%m/%d %H:%M:%S') for time in data['time']]
        except:
            timelist=[datetime.datetime.strptime(str(time),'%Y-%m-%d %H:%M:%S') for time in data['time']]
        # 默认第一点是没有问题的点
        for i in range(1, len(data)):  # 遍历各点
            timedist = (timelist[i]-timelist[i-1]).seconds
            d = haversine(lonlist[i - 1], latlist[i - 1], lonlist[i], latlist[i])
            if timedist > 1200:  # 如果时间差超过1200秒，则不进行修改
                continue
            
            if speed_iszero:
                speedlist[i] = d * 1.0 / timedist  ##使用经纬度更新速度
                
            if d > maxspeed * timedist:  # 异常距离 按照最大速度行驶也无法达到
                j = i#i为当前点
                #统计不正常点的数量
                while j + 1 < len(data) and d > maxspeed * timedist:#当前点不是最后一个点
                    j += 1#j指向后续点
                    d = haversine(lonlist[i - 1], latlist[i - 1], lonlist[j], latlist[j])
                    timedist = (timelist[j]-timelist[i]).seconds
                    if j - i > 10:  # 如果出现超出10个点不正常的情况，则放弃修改
                        j = i
                        break
                if j + 1 != len(data):
                    xDelta = (lonlist[j] - lonlist[i - 1]) / (j - i + 1)
                    yDelta = (latlist[j] - latlist[i - 1]) / (j - i + 1)
                    for ti in range(i, j):
                        lonlist[ti] = lonlist[ti - 1] + xDelta
                        latlist[ti] = latlist[ti - 1] + yDelta
                        deletedIndex.append(ti)
                else:
                    for ti in range(i, j):
                        lonlist[ti] = lonlist[ti - 1]
                        latlist[ti] = latlist[ti - 1]
                        deletedIndex.append(ti)
        
        # 默认第一点是没有问题的点
        # 针对放弃修改时 反向检索可能仍有需要修改的点
        for i in range(len(data) - 1, 1, -1):  # 遍历各点
            timedist = (timelist[i]-timelist[i-1]).seconds
            d = haversine(lonlist[i - 1], latlist[i - 1], lonlist[i], latlist[i])
            if timedist > 1200:  # 如果时间差超过1200秒，则不进行修改
                continue
            if speed_iszero:
                speedlist[i] = d * 1.0 / timedist  ##使用经纬度更新速度
            if d > maxspeed * timedist:  # 速度超出最大
                j = i
                while j + 1 < len(data) and d > maxspeed * timedist:
                    j += 1
                    d = haversine(lonlist[i - 1], latlist[i - 1], lonlist[j], latlist[j])
                    timedist = (timelist[j]-timelist[i-1]).seconds
                    if j - i > 10:  # 如果出现超出10个点不正常的情况，则放弃修改
                        j = i
                        break
                if j + 1 != len(data):
                    xDelta = (lonlist[j] - lonlist[i - 1]) / (j - i + 1)
                    yDelta = (latlist[j] - latlist[i - 1]) / (j - i + 1)
                    for ti in range(i, j):
                        lonlist[ti] = lonlist[ti - 1] + xDelta
                        latlist[ti] = latlist[ti - 1] + yDelta
                        deletedIndex.append(ti)
                else:
                    for ti in range(i, j):
                        lonlist[ti] = lonlist[ti - 1]
                        latlist[ti] = latlist[ti - 1]
                        deletedIndex.append(ti)
        deletedIndexs = []
        for i in deletedIndex:
            if i not in deletedIndexs:
                deletedIndexs.append(i)
        deletedIndexLabel = [data.index[i] for i in deletedIndexs]
        if len(deletedIndexLabel) > 0:
            data.drop(labels=deletedIndexLabel, inplace=True)
        
    return data

def cleandata():
    filenames=os.listdir('InitialData')
    for filename in filenames:
        # filename='wheat2'
        tablenames=os.listdir(os.path.join('InitialData',filename))
        for tablename in tablenames:
            path=os.path.join(filename,tablename)
            data=pd.read_excel(os.path.join('InitialData',path))
            # data=pd.read_excel('InitialData\\wheat1\\wheat_1_harvestor_1.xlsx')
            # data=pd.read_excel('InitialData\\wheat2\\wheat_2_harvestor_1.xlsx')
            # data=pd.read_excel('InitialData\\paddy\\paddy_harvestor_1.xlsx')
            # data=pd.read_excel('InitialData\\tractor\\tractor_1.xlsx')
            #列统一更名
            data.rename(columns={'longitude':'lon','经度':'lon','latitude':'lat','纬度':'lat','速度':'speed','方向':'dir','时间':'time','标签':'tags','标记':'tags'},inplace=True)
            data=data.loc[:,('time','lon','lat','speed','dir','tags')]
            data=data_pre_process(data)
            # visualize.draw_trajectory_picture(data)
            data.to_excel('CleanData/'+path)


if __name__ == "__main__":
    cleandata() 