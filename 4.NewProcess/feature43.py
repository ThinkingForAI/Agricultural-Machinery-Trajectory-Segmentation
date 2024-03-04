# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 00:52:59 2024

@author: DELL
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import math
from math import sqrt,sin,cos,asin,acos,radians
from sklearn.metrics import pairwise_distances_chunked
import datetime
import scipy
from scipy import stats

#经纬度几何距离   返回值：Euclidean距离
def geodistance(begin_lon_lat, end_lon_lat):
    lng1 = begin_lon_lat[0]
    lat1 = begin_lon_lat[1]
    lng2 = end_lon_lat[0]
    lat2 = end_lon_lat[1]
    distance = sqrt(
        pow((float(lng2) - float(lng1)), 2) +
        pow((float(lat2) - float(lat1)), 2))
    distance = round(distance, 10)
    return distance

#实际距离（m)   返回值：实际距离（m)
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


#除当前点的下一临近点外的 最近点和次近点  结果保存至文件   返回值：无
def getd1(data,filename,tablename):
    min_dist = []
    min_next_dist = []
    min_index = []
    min_next_index = []
    min_dist_info = []
    min_next_dist_info = []
    x=data.loc[:,('lon','lat')]
    
    #分块计算几何距离矩阵
    for chunked_results in pairwise_distances_chunked(x):
        for i in range(len(chunked_results)):
            dist_lst = list(enumerate(chunked_results[i]))
            if i == 0:
                dist_lst = dist_lst[2:]
            elif i == 1:
                dist_lst = dist_lst[3:]
            else:
                dist_lst = dist_lst[:i - 1] + dist_lst[i + 2:]
            new_dist_lst = sorted(dist_lst, key=lambda x: x[1])
            min_dist.append(new_dist_lst[0][1])
            min_index.append(new_dist_lst[0][0])
            min_next_dist.append(new_dist_lst[1][1])
            min_next_index.append(new_dist_lst[1][0])

    for i in range(len(min_dist)):
        min_dist_info.append([i, min_index[i], min_dist[i]])
        min_next_dist_info.append([i, min_next_index[i], min_next_dist[i]])

    df_min = pd.DataFrame(min_dist_info)
    df_min.columns = ["begin_point", "end_point", "distance"]
    df_min_next = pd.DataFrame(min_next_dist_info)
    df_min_next.columns = ["begin_point", "end_point", "distance"]
    #保存文件
    filepath1=os.path.join('Features43',filename,'点关系','first')
    os.makedirs(filepath1,exist_ok=True)
    #创建目录
    path1=os.path.join(filepath1,tablename)
    df_min.to_excel(path1)
    
    filepath2=os.path.join('Features43',filename,'点关系','second')
    #创建目录
    os.makedirs(filepath2,exist_ok=True)
    path2=os.path.join(filepath2,tablename)
    df_min_next.to_excel(path2)


#计算前向三个点后向三个点的几何距离 结果保存至文件 返回值：无
def gett12(data,filename,tablename):
    data_list_lon = data['lon'].values.flatten().tolist()
    data_list_lat = data['lat'].values.flatten().tolist()
    t_1andt_2begin_point = []
    t_1andt_2middle_point = []
    t_1andt_2end_point = []
    distance_begintomiddle = []
    distance_middletoend = []
    for j in range(len(data_list_lon)):
        begin_point = j
        if j != len(data_list_lon) - 2 and j != len(data_list_lon) - 1:
            middle_point = j + 1
            end_point = j + 2
            t_1andt_2begin_point.append(begin_point)
            t_1andt_2middle_point.append(middle_point)
            t_1andt_2end_point.append(end_point)
            d_begintomiddle = geodistance(
                [data_list_lon[begin_point], data_list_lat[begin_point]],
                [data_list_lon[middle_point], data_list_lat[middle_point]])
            d_middletoend = geodistance(
                [data_list_lon[middle_point], data_list_lat[middle_point]],
                [data_list_lon[end_point], data_list_lat[end_point]])
            distance_begintomiddle.append(d_begintomiddle)
            distance_middletoend.append(d_middletoend)
        elif j == len(data_list_lon) - 2:
            middle_point = j + 1
            end_point = 0
            t_1andt_2begin_point.append(begin_point)
            t_1andt_2middle_point.append(middle_point)
            t_1andt_2end_point.append(end_point)
            d_begintomiddle = geodistance(
                [data_list_lon[begin_point], data_list_lat[begin_point]],
                [data_list_lon[middle_point], data_list_lat[middle_point]])
            d_middletoend = 0
            distance_begintomiddle.append(d_begintomiddle)
            distance_middletoend.append(d_middletoend)
        else:
            middle_point = 0
            end_point = 0
            t_1andt_2begin_point.append(begin_point)
            t_1andt_2middle_point.append(middle_point)
            t_1andt_2end_point.append(end_point)
            d_begintomiddle = 0
            d_middletoend = 0
            distance_begintomiddle.append(d_begintomiddle)
            distance_middletoend.append(d_middletoend)
            
    all_oneandtwo_points_dis_to_excel = pd.DataFrame({
        'begin_point':
        t_1andt_2begin_point,
        'middle_point':
        t_1andt_2middle_point,
        'end_point':
        t_1andt_2end_point,
        'begintomiddle_distance':
        distance_begintomiddle,
        'middletoend_distance':
        distance_middletoend
    })
        
    d12,d23=distance_begintomiddle,distance_middletoend
    
    #保存文件
    filepath1=os.path.join('Features43',filename,'点关系','onedisandtwodis_down')
    os.makedirs(filepath1,exist_ok=True)
    path1=os.path.join(filepath1,tablename)
    all_oneandtwo_points_dis_to_excel.to_excel(path1)
    
    t_1andt_2begin_point = []
    t_1andt_2middle_point = []
    t_1andt_2end_point = []
    d_begintomiddle = 0
    d_middletoend = 0
    distance_begintomiddle = []
    distance_middletoend = []
    for j in range(len(data_list_lon)):
        begin_point = j
        if j != 0 and j != 1:
            middle_point = j - 1
            end_point = j - 2
            t_1andt_2begin_point.append(begin_point)
            t_1andt_2middle_point.append(middle_point)
            t_1andt_2end_point.append(end_point)
            d_begintomiddle = geodistance(
                [data_list_lon[begin_point], data_list_lat[begin_point]],
                [data_list_lon[middle_point], data_list_lat[middle_point]])
            d_middletoend = geodistance(
                [data_list_lon[middle_point], data_list_lat[middle_point]],
                [data_list_lon[end_point], data_list_lat[end_point]])
            distance_begintomiddle.append(d_begintomiddle)
            distance_middletoend.append(d_middletoend)
        elif j == 1:
            middle_point = j - 1
            end_point = 0
            t_1andt_2begin_point.append(begin_point)
            t_1andt_2middle_point.append(middle_point)
            t_1andt_2end_point.append(end_point)
            d_begintomiddle = geodistance(
                [data_list_lon[begin_point], data_list_lat[begin_point]],
                [data_list_lon[middle_point], data_list_lat[middle_point]])
            d_middletoend = 0
            distance_begintomiddle.append(d_begintomiddle)
            distance_middletoend.append(d_middletoend)
        else:
            middle_point = 0
            end_point = 0
            t_1andt_2begin_point.append(begin_point)
            t_1andt_2middle_point.append(middle_point)
            t_1andt_2end_point.append(end_point)
            d_begintomiddle = 0
            d_middletoend = 0
            distance_begintomiddle.append(d_begintomiddle)
            distance_middletoend.append(d_middletoend)

    all_oneandtwo_points_dis_to_excel = pd.DataFrame({
        'begin_point':
        t_1andt_2begin_point,
        'middle_point':
        t_1andt_2middle_point,
        'end_point':
        t_1andt_2end_point,
        'begintomiddle_distance':
        distance_begintomiddle,
        'middletoend_distance':
        distance_middletoend
    })
        
    d32,d21=distance_begintomiddle,distance_middletoend 
    
    filepath2=os.path.join('Features43',filename,'点关系','onedisandtwodis_up')
    os.makedirs(filepath2,exist_ok=True)
    path2=os.path.join(filepath2,tablename)
    all_oneandtwo_points_dis_to_excel.to_excel(path2)
    
    data['forward1_2'],data['forward2_3'],data['backward3_2'],data['backward2_1']=d12,d23,d32,d21
    
    
#计算经度、纬度、方向差值 不计高度 结果保存至文件 返回值：无
def cal_diff_main(data,filename,tablename):
    col_name = data.columns.tolist()
    col_name.insert(11, 'lon_diff')
    col_name.insert(12, "lat_diff")
    col_name.insert(13, "dir_diff")  #TODO
    data_list_lon = data['lon'].values.flatten().tolist(
    )  # flatten函数实现降维，将多维降成一维
    data_list_lat = data['lat'].values.flatten().tolist()
    data_list_dir = data['dir'].values.flatten().tolist()
    
    lon_diff = []
    lon_diff.append(0)
    lat_diff = []
    lat_diff.append(0)
    height_diff = []
    height_diff.append(0)
    dir_diff = []
    dir_diff.append(0)
    for i in range(len(data_list_lon)):
        if i == 0:
            continue
        else:
            lon_diff.append(
                float(data_list_lon[i]) - float(data_list_lon[i - 1]))
            lat_diff.append(
                float(data_list_lat[i]) - float(data_list_lat[i - 1]))
            dir_diff.append(
                float(data_list_dir[i]) - float(data_list_dir[i - 1]))
    data['lon_diff'] = lon_diff
    data['lat_diff'] = lat_diff
    data['dir_diff'] = dir_diff
    
    filepath2=os.path.join('Features43',filename,'差值计算')
    os.makedirs(filepath2,exist_ok=True)
    path2=os.path.join(filepath2,tablename)
    data.to_excel(path2)
    
    
#计算多个特征 距离段上的速度 加速度 2阶加速度 方位角变化 返回值：pandas对象传引用 所以不需返回值 data数据得到更改
def features_in_distance(data):
    data_list = data['time'].tolist()
    data_list_lon = data['lon'].tolist()
    data_list_lat = data['lat'].tolist()

    #统计每个距离段的实际长度（m）和时间间隔
    indistance=[]
    tindistance=[]
    for j in range(len(data_list)):
        if j==0:
            continue
        else:
            lon1,lat1,lon2,lat2=data_list_lon[j-1],data_list_lat[j-1],data_list_lon[j],data_list_lat[j]
            distance=haversine(lon1, lat1, lon2, lat2)
            try:
                d1 = datetime.datetime.strptime(str(data_list[j - 1]),
                                                '%Y/%m/%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(data_list[j]),
                                                '%Y/%m/%d %H:%M:%S')
            except:
                try:
                    d1 = datetime.datetime.strptime(str(data_list[j - 1]),
                                                    '%Y-%m-%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(data_list[j]),
                                                    '%Y-%m-%d %H:%M:%S')
                except:
                    d1 = datetime.datetime.strptime(str(data_list[j - 1]),
                                                    '%Y%m%d%H%M%S')
                    d2 = datetime.datetime.strptime(str(data_list[j]),
                                                    '%Y%m%d%H%M%S')
            indistance.append(distance)
            tindistance.append((d2-d1).seconds)

    # 平均速度
    average_velocity= []
    for j in range(len(data_list)):
        if j == 0:
            continue
        else:
            temp = tindistance[j-1]#时间
            distance=indistance[j-1]
            average_velocity.append(distance / temp)   #速度
    average_velocity.append(average_velocity[-1])#最近填充
    col_name = data.columns.tolist()
    col_name.insert(10, 'vind')
    data['vind'] = average_velocity
    
    # 平均加速度
    data_list_vin = data['vind'].values.flatten().tolist()
    average_acclec = []
    for j in range(len(data_list)):
        if j == 0:
            continue
        else:
            temp = tindistance[j-1]#时间
            average_acclec.append((data_list_vin[j] - data_list_vin[j - 1]) / temp)
    average_acclec.append(average_acclec[-1])
    col_name = data.columns.tolist()
    col_name.insert(11, 'accind')
    data['accind'] = average_acclec
    
    # 平均2阶加速度
    data_list_ap = data['accind'].values.flatten().tolist()
    average_2acclec = []
    for j in range(len(data_list)):
        if j == 0:
            continue
        else:
            temp = tindistance[j-1]
            average_2acclec.append((data_list_ap[j] - data_list_ap[j - 1]) / temp)
    average_2acclec.append(average_2acclec[-1])
    col_name = data.columns.tolist()
    col_name.insert(12, '2accind')
    data['2accind'] = average_2acclec
    
    # 方向角变化
    final_br = []
    bearings = []
    for j in range(len(data_list_lon)):
        if j == 0:
            continue
        else:
            lon1,lat1,lon2,lat2=data_list_lon[j-1],data_list_lat[j-1],data_list_lon[j],data_list_lat[j]
            y = (math.sin(lon2-lon1)) * (math.cos(lat2))
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2-lon1)
            bearings.append(math.atan2(y, x))
    for k in range(len(bearings)):
        if k == 0:
            continue
        else:
            final_br.append(bearings[k] - bearings[k - 1])
            
    final_br+=[0,0]
    col_name = data.columns.tolist()
    col_name.insert(13, 'BR')
    data['BR'] = final_br
    
    # 删去最后三个填充值
    length = len(data)
    result = data.iloc[0:len(data), :].copy()
    # 速度变为m/s的
    data['speed']=data['speed']/3.6
    
#滑动窗口标准差
def SD(num,lst):
    SD=[0 for x in range(len(lst))]
    SD[0]=np.std([lst[0]])
    for i in range(1,len(lst)):
        if(i<=num):
            SD[i]=np.std(lst[:i+1])
        else:
            SD[i] = np.std(lst[i-num:i+1])
    return SD

def sliding_window_std(num, lst):
    window_std = [0] * len(lst)  # 创建一个与 lst 长度相同的列表，用于存储滑动窗口内的标准差值

    for i in range(len(lst)):
        if i < num:
            window = lst[:i+1]  # 当前窗口包含的元素为从列表开头到当前索引
        else:
            window = lst[i-num+1:i+1]  # 当前窗口包含的元素为从当前索引向前数 num 个元素到当前索引
        window_std[i] = np.std(window)  # 计算当前窗口内的标准差，并将结果赋值给 window_std 列表的当前索
    return window_std

#滑动窗口平均值
def AVER(num,lst):
    AVER=[0 for x in range(len(lst))]
    AVER[0]=np.mean([lst[0]])
    for i in range(1,len(lst)):
        if(i<=num):
            AVER[i]=np.mean(lst[:i+1])
        else:
            AVER[i] = np.mean(lst[i-num:i+1])
    return AVER

def sliding_window_average(num, lst):
    window_avg = [0] * len(lst)  # 创建一个与 lst 长度相同的列表，用于存储滑动窗口内的平均值
    for i in range(len(lst)):
        if i < num:
            window = lst[:i+1]  # 当前窗口包含的元素为从列表开头到当前索引
        else:
            window = lst[i-num+1:i+1]  # 当前窗口包含的元素为从当前索引向前数 num 个元素到当前索引
        window_avg[i] = np.mean(window)  # 计算当前窗口内的平均值，并将结果赋值给 window_avg 列表的当前索引
    return window_avg

#滑动窗口中位数
def sliding_window_median(num, lst):
    window_med = [0] * len(lst)  # 创建一个与 lst 长度相同的列表，用于存储滑动窗口内的中位数值
    for i in range(len(lst)):
        if i < num:
            window = lst[:i+1]  # 当前窗口包含的元素为从列表开头到当前索引
        else:
            window = lst[i-num+1:i+1]  # 当前窗口包含的元素为从当前索引向前数 num 个元素到当前索引
        window_med[i] = np.median(window)  # 计算当前窗口内的中位数，并将结果赋值给 window_med 列表的当前索引
    return window_med

#滑动窗口最大值
def sliding_window_max(num, lst):
    window_max = [0] * len(lst)  # 创建一个与 lst 长度相同的列表，用于存储滑动窗口内的最大值
    for i in range(len(lst)):
        if i < num:
            window = lst[:i+1]  # 当前窗口包含的元素为从列表开头到当前索引
        else:
            window = lst[i-num+1:i+1]  # 当前窗口包含的元素为从当前索引向前数 num 个元素到当前索引
        window_max[i] = max(window)  # 计算当前窗口内的最大值，并将结果赋值给 window_max 列表的当前索引
    return window_max

#滑动窗口最小值
def sliding_window_min(num, lst):
    window_min = [0] * len(lst)  # 创建一个与 lst 长度相同的列表，用于存储滑动窗口内的最小值
    for i in range(len(lst)):
        if i < num:
            window = lst[:i+1]  # 当前窗口包含的元素为从列表开头到当前索引
        else:
            window = lst[i-num+1:i+1]  # 当前窗口包含的元素为从当前索引向前数 num 个元素到当前索引
        window_min[i] = min(window)  # 计算当前窗口内的最小值，并将结果赋值给 window_min 列表的当前索引
    return window_min

#滑动窗口偏度系数
def Skew(num,lst):
    Skew = [0 for x in range(len(lst))]
    Skew[0] = lst[0]
    for i in range(1, len(lst)):
        if (i <= num):
            Skew[i] = stats.skew(lst[:i + 1], bias=False)
        else:
            Skew[i] = stats.skew(lst[i - num:i + 1], bias=False)
    return Skew

def sliding_window_skew(num, lst):
    window_skew = [0] * len(lst)  # 创建一个与 lst 长度相同的列表，用于存储滑动窗口内的偏度
    for i in range(len(lst)):
        if i < num:
            window = lst[:i+1]  # 当前窗口包含的元素为从列表开头到当前索引
        else:
            window = lst[i-num+1:i+1]  # 当前窗口包含的元素为从当前索引向前数 num 个元素到当前索引
        window_skew[i] = stats.skew(window, bias=False)  # 计算当前窗口内的偏度，并将结果赋值给 window_skew 列表的当前索引
    return window_skew

#滑动窗口峰度系数
def Kurt(num,lst):
    Kurt = [0 for x in range(len(lst))]
    
    Kurt[0] = lst[0]
    for i in range(1, len(lst)):
        if (i <= num):
            Kurt[i] = stats.kurtosis(lst[:i + 1], bias=False)
        else:
            Kurt[i] = stats.kurtosis(lst[i - num:i + 1], bias=False)
    return Kurt


def sliding_window_kurt(num, lst):
    window_kurt = [0] * len(lst)  # 创建一个与 lst 长度相同的列表，用于存储滑动窗口内的峰度
    for i in range(len(lst)):
        if i < num:
            window = lst[:i+1]  # 当前窗口包含的元素为从列表开头到当前索引
        else:
            window = lst[i-num+1:i+1]  # 当前窗口包含的元素为从当前索引向前数 num 个元素到当前索引
        window_kurt[i] = stats.kurtosis(window, bias=False)  # 计算当前窗口内的峰度，并将结果赋值给 window_kurt 列表的当前索引
    return window_kurt

#滑动窗口变异系数
def Var (num,lst):
    Var = [0 for x in range(len(lst))]
    Var[0] = lst[0]
    for i in range(1, len(lst)):
        if (i <= num):
            Var[i] = scipy.stats.variation(lst[:i+1], axis=0)
        else:
            Var[i] = scipy.stats.variation(lst[i-num:i+1], axis=0)
    Var = np.array(Var)
    Var = np.nan_to_num(Var)
    Var = Var.tolist()
    return Var

def sliding_window_Var(num, lst):
    Var = [0 for x in range(len(lst))]  # 创建一个与 lst 长度相同的列表，用于存储滑动窗口内的变异系数
    Var[0] = lst[0]
    for i in range(1, len(lst)):
        if i <= num:
            Var[i] = scipy.stats.variation(lst[:i+1], axis=0)
        else:
            Var[i] = scipy.stats.variation(lst[i-num:i+1], axis=0)
    Var = np.array(Var)
    Var = np.nan_to_num(Var)
    Var = Var.tolist()
    return Var

#形成特征并保存文件 返回值：数据及标签
def features_add_and_save(path,filename,tablename):
    data=pd.read_excel(path,index_col=0)
    data.reset_index(drop=True,inplace=True)
    
    #根据算法需要读取清洗好数据的属性 
    clean_x = data['lon']
    clean_y = data['lat']
    clean_speed = data['speed']
    newrow_tag = data['tags']
    direct = data['dir']
    data_list=data['time']
    
    # 10个临近点距离之和
    distance=[0 for i in range(len(clean_x))]
    for i in range(0,len(clean_x)-10):
        dis = 0
        for j in range(i+1,i+11):
            temp=haversine(clean_x[i],clean_y[i],clean_x[j],clean_y[j])
            dis=dis+temp
        distance[i]=dis
    for i in range(len(clean_x)-10,len(clean_x)):
        dis = 0
        for j in range(i-1,i-11,-1):
            temp=haversine(clean_x[i],clean_y[i],clean_x[j],clean_y[j])
            dis = dis + temp
        distance[i]=dis
 
    # 10个临近点平均距离
    ave_distance=[0 for i in range(len(clean_x))]
    for i in range(0,len(clean_x)-10):
        dis = 0
        for j in range(i+1,i+11):
            temp=haversine(clean_x[i],clean_y[i],clean_x[j],clean_y[j])
            dis=dis+temp
        aver_dis=dis/10
        ave_distance[i]=aver_dis
    for i in range(len(clean_x)-10,len(clean_x)):
        dis = 0
        for j in range(i-1,i-11,-1):
            temp=haversine(clean_x[i],clean_y[i],clean_x[j],clean_y[j])
            dis = dis + temp
        aver_dis=dis/10
        ave_distance[i]=aver_dis
       
    #时间统计
    time=[0 for i in range(len(data_list))]
    for j in range(len(data_list)):
        if j==0:
            continue
        else:
            # lon1,lat1,lon2,lat2=data_list_lon[j-1],data_list_lat[j-1],data_list_lon[j],data_list_lat[j]
            # distance=haversine(lon1, lat1, lon2, lat2)
            try:
                d1 = datetime.datetime.strptime(str(data_list[j - 1]),
                                                '%Y/%m/%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(data_list[j]),
                                                '%Y/%m/%d %H:%M:%S')
            except:
                try:
                    d1 = datetime.datetime.strptime(str(data_list[j - 1]),
                                                    '%Y-%m-%d %H:%M:%S')
                    d2 = datetime.datetime.strptime(str(data_list[j]),
                                                    '%Y-%m-%d %H:%M:%S')
                except:
                    d1 = datetime.datetime.strptime(str(data_list[j - 1]),
                                                    '%Y%m%d%H%M%S')
                    d2 = datetime.datetime.strptime(str(data_list[j]),
                                                    '%Y%m%d%H%M%S')
            # indistance.append(distance)
            time[j]=(d2-d1).seconds
    
        
    # 加速度
    acc=[0 for i in range(len(clean_speed))]
    speed_diff=[0 for i in range(len(clean_speed))]
    for j in range(len(list(clean_speed))):
        if j == 0:
            speed_diff[j] = 0
            acc[j] = 0
        else:
            speed_diff[j] = clean_speed[j] - clean_speed[j - 1]
            acc[j] = float(speed_diff[j]) / time[j]
    
    # 角度差
    angular_diff=[0 for i in range(len(direct))]
    for j in range(len(list(clean_speed))):
        if j == 0:
            angular_diff[j] = 0
        else:
            angular_diff[j] = direct[j] - direct[j - 1]
    
    # 角速度
    angular_speed=[0 for i in range(len(direct))]
    for j in range(len(list(clean_speed))):
        if j == 0:
            angular_speed[j] = 0
        else:
            if clean_speed[j] == 0:
                angular_speed[j] = 0
            else:
                angular_speed[j] = direct[j] /  time[j]
    
    # 角加速度
    angular_acclec=[0 for i in range(len(direct))]
    angular_speed_diff=[0 for i in range(len(direct))]
    for j in range(len(list(clean_speed))):
        if j == 0:
            angular_speed_diff[j] = 0
            angular_acclec[j] = 0
        else:
            angular_speed_diff[j] = angular_speed[j] - angular_speed[j - 1]
            angular_acclec[j] = angular_speed_diff[j] /  time[j]
    
            
    cit1 = 100
    cit2 = 20
    # 标准差滑动窗口
    speed_SD = sliding_window_std(cit1, clean_speed)
    acc_SD = sliding_window_std(cit1, acc)
    angular_diff_SD = sliding_window_std(cit1, angular_diff)
    angular_speed_SD = sliding_window_std(cit1, angular_speed)
    angular_acclec_SD = sliding_window_std(cit1, angular_acclec)
    
    # 均值
    speed_AVER = sliding_window_average(cit1, clean_speed)
    acc_AVER = sliding_window_average(cit1, acc)
    angular_diff_AVER = sliding_window_average(cit1, angular_diff)
    angular_speed_AVER = sliding_window_average(cit1, angular_speed)
    angular_acclec_AVER = sliding_window_average(cit1, angular_acclec)
    
    # 中位数
    speed_med = sliding_window_median(cit1, clean_speed)
    acc_med = sliding_window_median(cit1, acc)
    angular_diff_med = sliding_window_median(cit1, angular_diff)
    angular_speed_med = sliding_window_median(cit1, angular_speed)
    angular_acclec_med = sliding_window_median(cit1, angular_acclec)
    
    # 最大值
    speed_max = sliding_window_max(cit1, clean_speed)
    acc_max = sliding_window_max(cit1, acc)
    angular_diff_max = sliding_window_max(cit1, angular_diff)
    angular_speed_max = sliding_window_max(cit1, angular_speed)
    angular_acclec_max = sliding_window_max(cit1, angular_acclec)
    
    # 最小值
    speed_min = sliding_window_min(cit1, clean_speed)
    acc_min = sliding_window_min(cit1, acc)
    angular_diff_min = sliding_window_min(cit1, angular_diff)
    angular_speed_min = sliding_window_min(cit1, angular_speed)
    angular_acclec_min = sliding_window_min(cit1, angular_acclec)
    
    # 变异系数
    speed_Var = sliding_window_Var(cit1, clean_speed)
    acc_Var = sliding_window_Var(cit1, acc)
    angular_diff_Var = sliding_window_Var(cit1, angular_diff)
    angular_speed_Var = sliding_window_Var(cit1, angular_speed)
    angular_acclec_Var = sliding_window_Var(cit1, angular_acclec)
    
    # 偏度
    speed_skew = sliding_window_skew(cit1, clean_speed)
    acc_skew = sliding_window_skew(cit1, acc)
    angular_diff_skew = sliding_window_skew(cit1, angular_diff)
    angular_speed_skew = sliding_window_skew(cit1, angular_speed)
    angular_acclec_skew = sliding_window_skew(cit1, angular_acclec)
    
    # 峰度
    speed_kurt = sliding_window_kurt(cit1, clean_speed)
    acc_kurt = sliding_window_kurt(cit1, acc)
    angular_diff_kurt = sliding_window_kurt(cit1, angular_diff)
    angular_speed_kurt = sliding_window_kurt(cit1, angular_speed)
    angular_acclec_kurt = sliding_window_kurt(cit1, angular_acclec)
    
    #基本特征已加入 速度 经度 维度 
    #添加计算基本特征 加速度 角度变化 角速度 角加速度 临近点距离和
    # 增加特征
    
    #前后连续三点
    gett12(data,filename,tablename)
    #差值计算
    cal_diff_main(data, filename, tablename)
    #距离段上的特征
    features_in_distance(data)#传引用
    
    data['distance']=distance
    data['acclec']=acc
    data['angular_diff']=angular_diff
    data['angular_speed']=angular_speed
    data['angular_acclec']=angular_acclec
    #添加标准差
    data['speed_SD'],data['acc_SD'],data['angular_diff_SD'],data['angular_speed_SD'],data['angular_acclec_SD']=speed_SD,acc_SD,angular_diff_SD,angular_speed_SD,angular_acclec_SD 
    #添加中位数
    data['speed_med'],data['acc_med'],data['angular_diff_med'],data['angular_speed_med'],data['angular_acclec_med']=speed_med,acc_med,angular_diff_med,angular_speed_med,angular_acclec_med
    #添加最大值
    data['speed_max'],data['acc_max'],data['angular_diff_max'],data['angular_speed_max'],data['angular_acclec_max']=speed_max,acc_max,angular_diff_max,angular_speed_max,angular_acclec_max
    #添加最小值
    data['speed_min'],data['acc_min'],data['angular_diff_min'],data['angular_speed_min'],data['angular_acclec_min']=speed_min,acc_min,angular_diff_min,angular_speed_min,angular_acclec_min
    
    # #添加偏度
    # data['speed_skew'],data['acc_skew'],data['angular_diff_skew'],data['angular_speed_skew'],data['angular_acclec_skew']=speed_skew,acc_skew,angular_diff_skew,angular_speed_skew,angular_acclec_skew
    # #添加峰度
    # data['speed_kurt'],data['acc_kurt'],data['angular_diff_kurt'],data['angular_speed_kurt'],data['angular_acclec_kurt']=speed_kurt,acc_kurt,angular_diff_kurt,angular_speed_kurt,angular_acclec_kurt
    
    #添加平均值
    data['speed_AVER'],data['acc_AVER'],data['angular_diff_AVER'],data['angular_speed_AVER'],data['angular_acclec_AVER']=speed_AVER,acc_AVER,angular_diff_AVER,angular_speed_AVER,angular_acclec_AVER
    
    #空值填充
    data.fillna(0)
    #标签保存 时间、标签删除
    labels=data['tags']
    data=data.drop(['tags','time'],axis=1)
    #方向删除
    data=data.drop(['dir'],axis=1)
    
    #添加方向
    data['dir']=direct
    #添加临近点平均距离
    data['average_distance']=ave_distance
    #添加变异系数
    data['speed_Var'],data['acc_Var'],data['angular_diff_Var'],data['angular_speed_Var'],data['angular_acclec_Var']=speed_Var,acc_Var,angular_diff_Var,angular_speed_Var,angular_acclec_Var
    
    #类型转换
    data['angular_diff']=data.angular_diff.astype(float)
    data['angular_diff_max']=data.angular_diff_max.astype(float)
    data['angular_diff_min']=data.angular_diff_min.astype(float)
    filepath=os.path.join('Features43',filename,'feature')
    os.makedirs(filepath,exist_ok=True)
    data['tags']=labels
    #结果保存至文件
    data.to_csv(os.path.join(filepath,tablename))
    return data

def pointRelation_add(path,filename,tablename):
    data=pd.read_csv(path,index_col=0)
    data.reset_index(drop=True,inplace=True)
    #点关系处理
    getd1(data,filename,tablename) 
    gett12(data,filename,tablename)

if __name__=='__main__':
    filenames=os.listdir('CleanData')
    for filename in filenames:
        tablenames=os.listdir(os.path.join('CleanData',filename))
        for tablename in tablenames:
            path=os.path.join('CleanData',filename,tablename)
            data1=features_add_and_save(path, filename, tablename)
            pointRelation_add(path, filename, tablename)
    
    # tablename='corn_harvestor_2.xlsx'
    # path=os.path.join('CleanData',filename,tablename)
    # data2=features_add_and_save(path, filename, tablename)
        
    data=data1.iloc[:,2:]
    # data=data2.iloc[:,2:]
    X,y=data.iloc[:,:-1].values,data.iloc[:,-1].values
    # #标签分布情况
    # plt.bar([0,1],[len(y)-sum(y),sum(y)])
    
    # from sklearn.model_selection import train_test_split
    # X_train,X_test,y_train,y_true=train_test_split(X,y,test_size=0.3)
    
    # from sklearn.preprocessing import MinMaxScaler,StandardScaler
    # scaler=MinMaxScaler()
    # scaler.fit(X_train)
    # X_train=scaler.transform(X_train)
    # X_test=scaler.transform(X_test)
     
    # from sklearn.tree import DecisionTreeClassifier
    # tree=DecisionTreeClassifier(max_depth=7)
    # tree.fit(X_train,y_train)
    # y_pred=tree.predict(X_test)
    
    # from sklearn.metrics import classification_report,confusion_matrix
    # print(classification_report(y_true, y_pred))
    # print(confusion_matrix(y_true, y_pred))
    # plt.plot(tree.feature_importances_)
    




