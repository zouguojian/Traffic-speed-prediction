# -- coding: utf-8 --
# python 3.6
import pandas as pd
import csv
import os
import datetime

dragon_stations=['G008564001000310010', 'G008564001000320010',
                 'G008564001000210010', 'G008564001000220010',
                 'G002064001000410010', 'G002064001000420010',
                 'G002064001000320010', 'G002064001000310010',
                 'G002064001000210010', 'G002064001000220010',
                 'G000664001001610010', 'G000664001001620010',
                 'G000664001001510010', 'G000664001001520010',
                 'G000664001004110010', 'G000664001004120010',
                 'G000664001001410010', 'G000664001001420010',
                 'G000664001001310010', 'G000664001001320010',
                 'G000664001004010010', 'G000664001004020010',
                 'G000664001003910010', 'G000664001003920010',
                 'G000664001001210010', 'G000664001001220010',
                 'G000664001000910010', 'G000664001000920010',
                 'G000664001000820010', 'G000664001000810010',
                 'G000664001003810010', 'G000664001003820010',
                 'G000664001001720010', 'G000664001001710010',
                 'G000664001000720010', 'G000664001000710010',
                 'G008564001000410010', 'G008564001000420010',
                 'G002064001000520010', 'G002064001000510010']

keys=['gantry_id', 'tian', 'xiaoshi', 'fenzhong', 'cheliuliang'] # keys
months=[-1,31,29,31,30,31,30,31,31,30,31,30,31] # -1 represents a sentinel
hours=24    # 24 h
minutes=60  # 60 minutes

# data1=pd.read_csv('dragon3.csv',encoding='gb2312')
# print(data1.values)

# data=pd.read_excel('龙门架编号和名称.xlsx')
# print(list(data.values[:,0]))

# 思路，先取出每个站点的所有数据，然后，按照时间的顺序遍历每个站点的数据，累积相加即可！！！

def read_source(file_paths, beg_month=6,end_month=9,year=2021,encoding='utf-8'):
    '''
    :param file_paths: list:[file1, file2,...], that is all paths
    :param beg_month: begin month
    :param end_month: end month
    :param year:
    :param encoding: decoding methods
    :return:
    '''
    for dragon_station in dragon_stations:
        print('the dragon_station name is: ',dragon_station)
        dragon_station_data_list=list()
        # used to store the DataFrame data of each station
        for in_file_path in file_paths:
            dragon_station_data=pd.read_csv(filepath_or_buffer=in_file_path,encoding=encoding)
            # read each station data
            dragon_station_data_list.append(dragon_station_data.loc[(dragon_station_data['gantry_id'] == dragon_station)])
            # use list to store each station data

        for month in range(beg_month,end_month):
            # to traverse the input months list
            for day in range(1, months[month]+1):
                # to traverse the input days of each month
                current_date=str(year)+'-'+(str(month) if month>9 else '0'+str(month)) +'-'+(str(day) if day>9 else '0'+str(day))
                for hour in range(hours):
                    for minute in range(0, minutes, 5):
                        sum_flow=0
                        for data in dragon_station_data_list: # read data form the DataFrom list
                            if not data.loc[(data['tian'] == current_date) & (data['xiaoshi'] == hour) & (data['fenzhong'] == minute)].empty:
                                sum_flow+=int(data.loc[(data['tian'] == current_date) & (data['xiaoshi'] == hour) & (data['fenzhong'] == minute)].values[-1][-1])
                        print(dragon_station, current_date,hour,minute,sum_flow)
                        yield dragon_station, current_date.replace('-','/'),hour,minute,sum_flow

def data_combine(file_paths, out_path, beg_month=6,end_month=9,year=2021,encoding='utf-8'):
    '''
    :param file_paths: a list, contain original data sets
    :param out_path: write path, used to save the training set
    :return:
    '''
    file = open(out_path, 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(['station','date','hour','minute','flow'])
    for line in read_source(file_paths=file_paths, beg_month=beg_month,end_month=end_month, year=year, encoding=encoding):
        writer.writerow(line)
    file.close()

if __name__=='__main__':
    print('beginning')
    # data_combine(file_paths=['dragon1.5.csv', 'dragon1.csv', 'dragon2.csv', 'dragon3.csv'], out_path='dragon_flow.csv', encoding='gb2312')
    # read_source(file_paths=['in1.5.csv', 'in1.csv', 'in2.csv', 'in3.csv'], year=2021, encoding='gb2312')
    # for line in read_source(file_paths=['in1.5.csv','in1.csv','in2.csv','in3.csv'],year=2021,encoding='gb2312'):
    #     print(line)

    # data1=pd.read_csv('dragon_flow.csv',encoding='utf-8')
    data = pd.read_csv('dragon_flow_fill_new.csv', encoding='utf-8').values


    # file = open('dragon_flow_fill_new.csv', 'w', encoding='utf-8')
    # writer = csv.writer(file)
    # writer.writerow(['station','date','hour','minute','flow'])
    # for line in data:
    #     line[2]=line[2].replace('-','/')
    #     writer.writerow(line[1:])
    # file.close()

    #
    # print(data1.values)
    print(data.shape)
    #
    # for i in range(data.shape[0]):
    #     print(data1[i])
    #     print(data[i])


    # zero_index=0
    # start_date='2021/6/1'
    # for line in data1.values:
    #     if line[-1]==0:
    #         zero_index+=1
    #         if zero_index==1:
    #             start_date=line[1]
    #     else:
    #         if zero_index>=120:
    #             print(line[0],start_date,line[1])
    #             zero_index=0
    #         else:
    #             zero_index=0
    #             start_date=line[1]


    print('finished')