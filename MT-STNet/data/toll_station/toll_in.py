# -- coding: utf-8 --
# python 3.6
import pandas as pd
import csv
import os
import datetime

in_toll_stations=[2002,2005,2007,2008,2009,2011,2012,101001,101007,102004,102005,106006,106007]
keys=['入口站点', '日期', '小时', '分钟', '车流量'] # keys
months=[-1,31,29,31,30,31,30,31,31,30,31,30,31] # -1 represents a sentinel
hours=24    # 24 h
minutes=60  # 60 minutes

# data1=pd.read_csv('in1.5.csv',encoding='gb2312')
# print(data1.values[:10])
# print(data1.loc[(data1['日期']=='2021/6/1')])

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
    for in_toll_station in in_toll_stations:
        print('the in_toll_station name is: ',in_toll_station)
        in_toll_station_data_list=list()
        # used to store the DataFrame data of each station
        for in_file_path in file_paths:
            in_toll_station_data=pd.read_csv(filepath_or_buffer=in_file_path,encoding=encoding)
            # read each station data
            in_toll_station_data_list.append(in_toll_station_data.loc[(in_toll_station_data['入口站点'] == in_toll_station)])
            # use list to store each station data

        for month in range(beg_month,end_month):
            # to traverse the input months list
            for day in range(1, months[month]+1):
                # to traverse the input days of each month
                current_date=str(year)+'/'+str(month)+'/'+str(day)
                for hour in range(hours):
                    for minute in range(0, minutes, 5):
                        sum_flow=0
                        for data in in_toll_station_data_list: # read data form the DataFrom list
                            if not data.loc[(data['日期'] == current_date) & (data['小时'] == hour) & (data['分钟'] == minute)].empty:
                                sum_flow+=int(data.loc[(data['日期'] == current_date) & (data['小时'] == hour) & (data['分钟'] == minute)].values[-1][-1])
                        print(in_toll_station, current_date,hour,minute,sum_flow)
                        yield in_toll_station, current_date,hour,minute,sum_flow

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
    print('hello')
    # data_combine(file_paths=['in1.5.csv', 'in1.csv', 'in2.csv', 'in3.csv'], out_path='in_flow.csv', encoding='gb2312')
    # read_source(file_paths=['in1.5.csv', 'in1.csv', 'in2.csv', 'in3.csv'], year=2021, encoding='gb2312')
    # for line in read_source(file_paths=['in1.5.csv','in1.csv','in2.csv','in3.csv'],year=2021,encoding='gb2312'):
    #     print(line)

    # data = pd.read_csv('in_flow.csv', encoding='utf-8').values

    # file = open('in_flow_train.csv', 'w', encoding='utf-8')
    # writer = csv.writer(file)
    # writer.writerow(['station','date','hour','minute','flow'])
    # for line in data:
    #     writer.writerow([str(line[0]),str(line[1]),int(line[2]),int(line[3]),int(line[4])])
    # file.close()

    print('finished')