# -- coding: utf-8 --
# python 3.6
import pandas as pd
import csv
import os
import datetime

in_toll_stations={'2002':0, '2005':1, '2007':2, '2008':3, '2009':4, '2011':5, '2012':6,
                  '101001':7, '101007':8, '102004':9, '102005':10, '106006':11 ,'106007':12}

out_toll_stations={'2002':13, '2005':14, '2007':15, '2008':16, '2009':17, '2011':18, '2012':19,
                  '101001':20, '101007':21, '102004':22, '102005':23, '106006':24 ,'106007':25}

dragon_stations={'G008564001000310010':26, 'G008564001000320010':27,
                 'G008564001000210010':28, 'G008564001000220010':29,
                 'G002064001000410010':30, 'G002064001000420010':31,
                 'G002064001000320010':32, 'G002064001000310010':33,
                 'G002064001000210010':34, 'G002064001000220010':35,
                 'G000664001001610010':36, 'G000664001001620010':37,
                 'G000664001001510010':38, 'G000664001001520010':39,
                 'G000664001004110010':40, 'G000664001004120010':41,
                 'G000664001001410010':42, 'G000664001001420010':43,
                 'G000664001001310010':44, 'G000664001001320010':45,
                 'G000664001004010010':46, 'G000664001004020010':47,
                 'G000664001003910010':48, 'G000664001003920010':49,
                 'G000664001001210010':50, 'G000664001001220010':51,
                 'G000664001000910010':52, 'G000664001000920010':53,
                 'G000664001000820010':54, 'G000664001000810010':55,
                 'G000664001003810010':56, 'G000664001003820010':57,
                 'G000664001001720010':58, 'G000664001001710010':59,
                 'G000664001000720010':60, 'G000664001000710010':61,
                 'G008564001000410010':62, 'G008564001000420010':63,
                 'G002064001000520010':64, 'G002064001000510010':65}

file_path_types={'toll_station/in_flow.csv':in_toll_stations,
                 'toll_station/out_flow.csv':out_toll_stations,
                 'dragon_station/dragon_flow.csv':dragon_stations}

keys=['station','date','hour','minute','flow'] # keys
months=[-1,31,29,31,30,31,30,31,31,30,31,30,31] # -1 represents a sentinel
hours=24    # 24 h
minutes=60  # 60 minutes

# data1=pd.read_csv('in1.5.csv',encoding='gb2312')
# print(data1.values[:10])
# print(data1.loc[(data1['日期']=='2021/6/1')])

# 思路，按照每个站点的实际index进行排序
# each sample shape is : [indexs, features]

def data_train(file_paths, out_path,encoding='utf-8'):
    '''
    :param file_paths: a list, contain original data sets
    :param out_path: write path, used to save the training set
    :return:
    '''
    file = open(out_path, 'w', encoding=encoding)
    writer = csv.writer(file)
    writer.writerow(['station','date','day','hour','minute','flow'])
    station_data_dict=dict()
    rows=26496 # (30+31+31)*24*12

    # 读取所有station的values，并且按照station的映射id进行字典存储
    for path in file_paths:
        data = pd.read_csv(path, encoding=encoding)
        data['station']=data['station'].astype(str) # 这一步将station中的所有内通转化为字符串形式，不然下面代码会出现整数型数值
        for station in file_path_types[path].keys():
            station_data_dict[file_path_types[path][station]]=data.loc[(data['station'] == station)].values
            rows=data.loc[data['station'] == station].shape[0]

    # 顺序将station的值进行存储
    print(rows)
    for i in range(rows):
        print('the line index is : ', i)
        for station_index in station_data_dict.keys():
            writer.writerow([str(station_index)]+[station_data_dict[station_index][i][1]]+
                            [int(station_data_dict[station_index][i][1].split('/')[-1])]+list(station_data_dict[station_index][i])[2:])
    file.close()

if __name__=='__main__':
    print('hello')
    data_train(file_paths=['toll_station/in_flow.csv', 'toll_station/out_flow.csv', 'dragon_station/dragon_flow.csv'], out_path='train.csv', encoding='utf-8')

    print(pd.read_csv('train.csv',encoding='utf-8').shape)

    print('finished')
