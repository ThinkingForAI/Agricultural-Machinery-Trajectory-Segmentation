import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle

# 加载数据
# kfold = 5  # 交叉验证次数
allfile = []  # 文件夹下的所有文件
print('loading data...')
path='tractor_25_10fold' #分成10折所在的文件夹
all_path="tractor_25" #总文件所在文件夹
print('USE data:', path)
files = os.listdir(path)  # 得到文件夹下的所有文件名称
for filepath, dirnames, filenames in os.walk(all_path):
    for filename in filenames:
        if filename not in allfile:
            allfile.append(filename)
f_pred=[]
f_true=[]
acc_list=[]
def train():
    #内存计算方式 预先将所有数据读入
    data={}
    for table in allfile:
        data[table]=pd.read_excel(all_path+'/'+table)
    #初始化
    test_road_precision = 0
    test_road_recall = 0
    test_road_f1score = 0
    test_field_precision = 0
    test_field_recall = 0
    test_field_f1score = 0
    test_accuracy = 0
    test_macro_precision = 0
    test_macro_recall = 0
    test_macro_f1score = 0
    test_weight_precision = 0
    test_weight_recall = 0
    test_weight_f1score = 0
    kfold=10
    lenset=kfold
    final_indice = []
    final_y = []
    for fold in range(kfold):
        print('Kfold: ',fold)
        trainfeature = pd.DataFrame()
        train_tag = pd.DataFrame()
        testfeature = pd.DataFrame()    
        test_tag = pd.DataFrame()
        validfeature = pd.DataFrame()
        valid_tag = pd.DataFrame()
        # train_vid, testvid, _, _ = train_test_split(allfile, allfile, test_size=0.2, random_state=fold)
        # train_vid, validvid, _, _ = train_test_split(train_vid, train_vid, test_size=1/6, random_state=fold)
        train_vid=os.listdir(path+'/'+str(fold)+"/train")
        validvid=os.listdir(path+'/'+str(fold)+"/val")
        testvid = os.listdir(path +'/'+ str(fold) + "/test")
        #显示轨迹数量
        print('len train_vid:',len(train_vid))
        print('len validvid:', len(validvid))
        print('len testvid:',len(testvid))
        for i in range(len(allfile)):
            #print(allfile[i])
            data_loc=data[allfile[i]]
            tag=data_loc['tag']
            data_loc=data_loc.drop(['tag','lon','lat'],axis=1)
            if allfile[i] in testvid:
                testfeature=pd.concat([testfeature,data_loc])
                test_tag=pd.concat([test_tag,tag])
            elif allfile[i] in train_vid:
                trainfeature=pd.concat([trainfeature,data_loc])
                train_tag=pd.concat([train_tag,tag])
            else:
                validfeature=pd.concat([validfeature,data_loc])
                valid_tag=pd.concat([valid_tag,tag]) 
        print("testfeature:",len(testfeature))
        print("validfeature:",len(validfeature))
        print("trainfeature:",len(trainfeature))
        
        # break
        #最佳参数选择
        trainacc = -1
        validacc = -1
        testacc = -1
        map_criterion_se = None
        map_min_samples_leaf_se =None
        map_splitter =None
        for criterion_se in ['entropy', 'gini']:
            for min_samples_leaf_se in [2,8, 16, 32, 64, 128,256,512]:
                for splitter_se in ['best','random']:

                    clf = DecisionTreeClassifier(random_state=fold, criterion= criterion_se, min_samples_leaf= min_samples_leaf_se, splitter = splitter_se)

                    clf.fit(trainfeature, train_tag)

                    pred = clf.predict(validfeature)

                    temp_acc = accuracy_score(valid_tag, pred)
                    if temp_acc > validacc:
                        validacc = temp_acc
                        map_criterion_se = criterion_se
                        map_min_samples_leaf_se = min_samples_leaf_se
                        map_splitter = splitter_se
        #输出最佳参数
        print('valid acc: ',validacc)
        print('map_criterion_se: ',map_criterion_se)
        print('map_min_samples_leaf_se: ', map_min_samples_leaf_se)
        print('map_splitter: ', map_splitter)
        #entropy 256 best
        # map_criterion_se='entropy' gini
        # map_min_samples_leaf_se=256
        # map_splitter=best
        
        #利用最佳参数训练模型
        bestclf = DecisionTreeClassifier(random_state=fold, criterion= map_criterion_se, min_samples_leaf= map_min_samples_leaf_se, splitter = map_splitter)
        bestclf.fit(trainfeature, train_tag)
        pred = bestclf.predict(testfeature)
        test_tag=test_tag.astype(int)
        valid_tag=valid_tag.astype(int) 
        
        final_y.append(test_tag)
        final_indice.append(pred)
        f_true.append(test_tag)
        f_pred.append(pred)
        y_final = np.concatenate(final_y)
        indice_final = np.concatenate(final_indice)
        # print(classification_report(test_tag, pred, digits=4))
        
        acc_list.append(classification_report(y_final, indice_final, digits=4, output_dict=True)['accuracy'])
        test_road_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['0'][
            'precision']
        test_road_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['0']['recall']
        test_road_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['0']['f1-score']
        test_field_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['1'][
            'precision']
        test_field_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['1']['recall']
        test_field_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['1']['f1-score']
        test_accuracy += classification_report(y_final, indice_final, digits=4, output_dict=True)['accuracy']
        test_macro_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg'][
            'precision']
        test_macro_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg'][
            'recall']
        test_macro_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg'][
            'f1-score']
        test_weight_precision += \
        classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg'][
            'precision']
        test_weight_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg'][
            'recall']
        test_weight_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg'][
            'f1-score']

        final_y = []
        final_indice = []
    # y_final = np.concatenate(f_true)
    # indice_final = np.concatenate(f_pred)
    # print('test acc: ', accuracy_score(y_final, indice_final))
    # print(classification_report(y_final, indice_final, digits=4))
    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('               precision    recall    f1score')
    print('        0    ' + str(round(test_road_precision / lenset, 4)) + '      ' + str(
        round(test_road_recall / lenset, 4)) + '    ' + str(round(test_road_f1score / lenset, 4)))
    print('        1    ' + str(round(test_field_precision / lenset, 4)) + '      ' + str(
        round(test_field_recall / lenset, 4)) + '    ' + str(round(test_field_f1score / lenset, 4)))
    print('\n')
    print('    accuracy                          ' + str(round(test_accuracy / lenset, 4)))
    print('   macro avg    ' + str(round(test_macro_precision / lenset, 4)) + '      ' + str(
        round(test_macro_recall / lenset, 4)) + '    ' + str(round(test_macro_f1score / lenset, 4)))
    print('weighted avg    ' + str(round(test_weight_precision / lenset, 4)) + '      ' + str(
        round(test_weight_recall / lenset, 4)) + '    ' + str(round(test_weight_f1score / lenset, 4)))
# def pre_true_ground():
#     for fold in range(kfold):
#         print('Kfold: ',fold)
#         train_vid, testvid, _, _ = train_test_split(allfile, allfile, test_size=0.23, random_state=600)
#         train_vid, validvid, _, _ = train_test_split(train_vid, train_vid, test_size=0.2, random_state=600)
#         map_criterion_se = 'gini'
#         map_min_samples_leaf_se = 256
#         map_splitter = 'best'
#         bestclf = DecisionTreeClassifier(random_state=600, criterion=map_criterion_se,
#                                          min_samples_leaf=map_min_samples_leaf_se, splitter=map_splitter)
#         trainfeature = []
#         train_tag = []
#         test_vid_all=[]
#         for i in range(len(allfile)):
#             data_loc =[]
#             data_path = path + allfile[i]
#             data_df = pd.read_csv(data_path)
#             lon = data_df['lon_diff'].tolist()
#             data_loc.append(lon)
#             lat = data_df['lat_diff'].tolist()
#             data_loc.append(lat)
#             dir = data_df['dir_diff'].tolist()
#             data_loc.append(dir)
#             # height = data_df['height'].tolist()
#             # data_loc.append(height)
#             vin = data_df['vin'].tolist()
#             data_loc.append(vin)
#             speed = data_df['speed'].tolist()
#             data_loc.append(speed)
#             ap = data_df['ap'].tolist()
#             data_loc.append(ap)
#             jp = data_df['jp'].tolist()
#             data_loc.append(jp)
#             br = data_df['br'].tolist()
#             data_loc.append(br)
#             #seg_num = data_df['seg_num'].tolist()
#             tag = data_df['tag'].tolist()
#             data_loc = np.array(data_loc)
#             data_loc = data_loc.transpose()
#             if allfile[i] in testvid:
#                 test_vid_all.append(allfile[i])
#             if allfile[i] in train_vid:
#                 for j in range(len(tag)):
#                     trainfeature.append(data_loc[j])
#                     train_tag.append(tag[j])
#         bestclf.fit(trainfeature, train_tag)
#         for j in range(len(test_vid_all)):
#             print(test_vid_all[j])
#             testfeature = []
#             test_tag = []
#             data_loc = []
#             data_path = path + test_vid_all[j]
#             data_df = pd.read_csv(data_path)
#             print(len(data_df['lon_diff']))
#             lon = data_df['lon_diff'].tolist()
#             data_loc.append(lon)
#             lat = data_df['lat_diff'].tolist()
#             data_loc.append(lat)
#             dir = data_df['dir_diff'].tolist()
#             data_loc.append(dir)
#             # height = data_df['height'].tolist()
#             # data_loc.append(height)
#             vin = data_df['vin'].tolist()
#             data_loc.append(vin)
#             speed = data_df['speed'].tolist()
#             data_loc.append(speed)
#             ap = data_df['ap'].tolist()
#             data_loc.append(ap)
#             jp = data_df['jp'].tolist()
#             data_loc.append(jp)
#             br = data_df['br'].tolist()
#             data_loc.append(br)
#             # seg_num = data_df['seg_num'].tolist()
#             tag = data_df['tag'].tolist()
#             data_loc = np.array(data_loc)
#             data_loc = data_loc.transpose()
#             for k in range(len(tag)):
#                 testfeature.append(data_loc[k])
#                 test_tag.append(tag[k])
#             pred = bestclf.predict(testfeature)

#             print(classification_report(test_tag, pred, digits=4))
#             indice_final = list(pred)
#             indice_excel = pd.DataFrame(indice_final)
#             indice_excel.columns = ['tag']
#             # file = open(, "w")
#             indice_excel.to_csv('/home/liguangyuan/GCN_system/all_result/小麦/DT/' + test_vid_all[j], index=False)
#             print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
if __name__=="__main__":
    train()
    print(acc_list)
    #pre_true_ground()
    
#绘制测试集标签分布
y_road=[]
y_field=[]
for i in range(kfold):
    testdata=pd.DataFrame()
    for j in os.listdir(path+'/'+str(i)+'/'+'test'):
        tmp=testdata[j]
        testdata=pd.concat((testdata,tmp))
    y_road.append(testdata.groupby('tag').describe()[('speed','count')][0])
    y_field.append(testdata.groupby('tag').describe()[('speed','count')][1])
x=np.arange(1,kfold+1)
plt.bar(x - 0.25/2, y_road, 0.25, label='road')
plt.bar(x + 0.25/2, y_field, 0.25, label='field')
plt.xticks(x)
plt.ylabel('Numbers of Points')
plt.title('Label Distribution')
plt.legend()
plt.show()


