import os
import datetime
import pandas
import pandas as pd
import numpy as np
cit1=5
cit2=50

def get_5_feature(data):
    lon_list = data['lon']
    lat_list = data['lat']
    speed_list = data['speed']
    angular_list = data['dir']
    time_list = data['time']
    tag_list = data['tags']
    #统计特征计算
    speed_diff = [0 for x in range(len(speed_list))]
    time_diff = [0 for x in range(len(speed_list))]
    acclec = [0 for x in range(len(speed_list))]
    angular_speed_diff = [0 for x in range(len(speed_list))]
    angular_acclec = [0 for x in range(len(speed_list))]
    angular_diff = [0 for x in range(len(speed_list))]
    angular_speed = [0 for x in range(len(speed_list))]
    for j in range(len(list(speed_list))):
        if j == 0:
            speed_diff[j] = 0
            time_diff[j] = 0
            acclec[j] = 0
        else:
            speed_diff[j] = speed_list[j] - speed_list[j - 1]
            try:
                d1 = datetime.datetime.strptime(str(time_list[j - 1]), '%Y/%m/%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(time_list[j]), '%Y/%m/%d %H:%M:%S')
            except:
                d1 = datetime.datetime.strptime(str(time_list[j - 1]), '%Y-%m-%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(time_list[j]), '%Y-%m-%d %H:%M:%S')
            time_diff[j] = (d2 - d1).seconds
            acclec[j] = float(speed_diff[j]) / time_diff[j]
    for temp in range(len(list(speed_list))):
        if temp == 0:
            angular_speed[temp] = 0
        else:
            if speed_list[temp] == 0:
                angular_speed[temp] = 0
            else:
                angular_speed[temp] = angular_list[temp] / time_diff[temp]
    for k in range(len(list(speed_list))):
        if k == 0:
            angular_speed_diff[k] = 0
            time_diff[k] = 0
            angular_acclec[k] = 0
        else:
            angular_speed_diff[k] = angular_speed[k] - angular_speed[k - 1]
            try:
                d1 = datetime.datetime.strptime(str(time_list[k - 1]), '%Y/%m/%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(time_list[k]), '%Y/%m/%d %H:%M:%S')
            except:
                d1 = datetime.datetime.strptime(str(time_list[k - 1]), '%Y-%m-%d %H:%M:%S')
                d2 = datetime.datetime.strptime(str(time_list[k]), '%Y-%m-%d %H:%M:%S')
            time_diff[k] = (d2 - d1).seconds
            angular_acclec[k] = angular_speed_diff[k] / time_diff[k]
    for l in range(len(list(speed_list))):
        if l == 0:
            angular_diff[l] = 0
        else:
            angular_diff[l] = angular_list[l] - angular_list[l - 1]
    data['acceleration'] = acclec
    data['angular_speed'] = angular_speed
    data['angular_acceleration'] = angular_acclec
    data['angle_diff'] = angular_diff

def get_median(data):
   data = sorted(data)
   size = len(data)
   if size % 2 == 0: # 判断列表长度为偶数
    #median = (data[size//2]+data[size//2-1])/2
    median =  data[size//2]
    data[0] = median
   if size % 2 == 1: # 判断列表长度为奇数
    median = data[(size-1)//2]
    data[0] = median
   return data[0]

def med(num,lst):
    med=[0 for x in range(len(lst))]
    med[0]=lst[0]
    for i in range(1,len(lst)):
        if(i<=num):
            med[i]=get_median(lst[:i+1])
        else:
            med[i] = get_median(lst[i-num:i+1])
    return med

def SD(num,lst):
    SD=[0 for x in range(len(lst))]
    SD[0]=np.std([lst[0]])
    for i in range(1,len(lst)):
        if(i<=num):
            SD[i]=np.std(lst[:i+1])
        else:
            SD[i] = np.std(lst[i-num:i+1])
    return SD

def features_add_and_save(path,savepath):
    files=os.listdir(path)
    for i in files:
        data=pd.read_excel(path+"/"+i,index_col=0)
        data.reset_index(drop=True,inplace=True)
        get_5_feature(data)
        
        tag_list=data["tags"]
        lon_list = data['lon']
        lat_list = data['lat']

        speed_list=data['speed']
        speed_med_5=med(cit1,speed_list)
        speed_med_20=med(cit2,speed_list)
        speed_SD_5=SD(cit1,speed_list)
        speed_SD_20 = SD(cit2, speed_list)

        acc_list = data['acceleration']
        acc_med_5 = med(cit1, acc_list)
        acc_med_20 = med(cit2, acc_list)
        acc_SD_5 = SD(cit1, acc_list)
        acc_SD_20 = SD(cit2, acc_list)

        ang_speed_list = data['angular_speed']
        ang_speed_med_5 = med(cit1, ang_speed_list)
        ang_speed_med_20 = med(cit2, ang_speed_list)
        ang_speed_SD_5 = SD(cit1, ang_speed_list)
        ang_speed_SD_20 = SD(cit2, ang_speed_list)

        ang_acc_list = data['angular_acceleration']
        ang_acc_med_5 = med(cit1, ang_acc_list)
        ang_acc_med_20 = med(cit2, ang_acc_list)
        ang_acc_SD_5 = SD(cit1, ang_acc_list)
        ang_acc_SD_20 = SD(cit2, ang_acc_list)

        ang_diff_list = data['angle_diff']
        ang_diff_med_5 = med(cit1, ang_diff_list)
        ang_diff_med_20 = med(cit2, ang_diff_list)
        ang_diff_SD_5 = SD(cit1, ang_diff_list)
        ang_diff_SD_20 = SD(cit2, ang_diff_list)
        
        data["speed_med_5"] = speed_med_5
        data["speed_med_20"] = speed_med_20
        data["speed_SD_5"] = speed_SD_5
        data["speed_SD_20"] = speed_SD_20

        data['acceleration_med_5'] = acc_med_5
        data['acceleration_med_20'] = acc_med_20
        data['acceleration_SD_5'] = acc_SD_5
        data['acceleration_SD_20'] = acc_SD_20

        data['angular_speed_med_5'] = ang_speed_med_5
        data['angular_speed_med_20'] = ang_speed_med_20
        data['angular_speed_SD_5'] = ang_speed_SD_5
        data['angular_speed_SD_20'] = ang_speed_SD_20

        data['angular_acceleration_med_5'] = ang_acc_med_5
        data['angular_acceleration_med_20'] = ang_acc_med_20
        data['angular_acceleration_SD_5'] = ang_acc_SD_5
        data['angular_acceleration_SD_20'] = ang_acc_SD_20

        data['angle_diff'] = ang_diff_list
        data['angle_diff_med_5'] = ang_diff_med_5
        data['angle_diff_med_20'] = ang_diff_med_20
        data['angle_diff_SD_5'] = ang_diff_SD_5
        data['angle_diff_SD_20'] = ang_diff_SD_20
        
        final_path=os.path.join(filepath,i)
        
        #最终数据保存时不在带上索引
        data.drop(['time'],axis=1).to_excel(final_path,index=False)

if __name__=="__main__":
    filenames=os.listdir('CleanData')
    for filename in filenames:
        path=os.path.join('CleanData',filename)
        #创造目录
        filepath=os.path.join('Features25',filename)
        os.makedirs(filepath,exist_ok=True)
        features_add_and_save(path,filepath)


