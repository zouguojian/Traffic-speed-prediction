# -- coding: utf-8 --
# python 3.6
import pandas as pd
import csv
import numpy as np
import os
import datetime

toll_dragon={('101001', '78001D'): 0, ('101001', '780079'): 1, ('101001', '79001C'): 2, ('101007', '78001F'): 3,
               ('101007', '79001E'): 4, ('102004', '780067'): 5, ('102004', '78007D'): 6, ('102004', '79007C'): 7,
               ('102005', '78007F'): 8, ('102005', '79007E'): 9, ('106006', '780069'): 10, ('106006', '790068'): 11,
               ('106007', '78006B'): 12, ('106007', '79006A'): 13, ('2002', '780011'): 14, ('2002', '790010'): 15,
               ('2005', '780023'): 16, ('2005', '790022'): 17, ('2007', '78001B'): 18, ('2007', '79001A'): 19,
               ('2008', '780019'): 20, ('2008', '790014'): 21, ('2009', '780019'): 22, ('2009', '790014'): 23,
               ('2011', '78005F'): 24, ('2011', '79005E'): 25, ('2012', '780061'): 26, ('2012', '790060'): 27}
dragon_dragon={('78000F', '780011'): 28, ('780011', '780013'): 29, ('780011', '78005D'): 30, ('780013', '780019'): 31,
               ('780019', '78001B'): 32, ('78001B', '78001D'): 33, ('78001B', '780079'): 34, ('78001D', '78001F'): 35,
               ('78001F', '780021'): 36, ('780021', '780023'): 37, ('78005D', '78005F'): 38, ('78005F', '780061'): 39,
               ('780061', '780063'): 40, ('780061', '78007B'): 41, ('780061', '79007A'): 42, ('780063', '780021'): 43,
               ('780067', '780069'): 44, ('780069', '78006B'): 45, ('780079', '780063'): 46, ('780079', '78007B'): 47,
               ('780079', '790062'): 48, ('78007B', '780067'): 49, ('78007B', '78007D'): 50, ('78007D', '78007F'): 51,
               ('790012', '790010'): 52, ('790014', '790012'): 53, ('79001A', '790014'): 54, ('79001C', '79001A'): 55,
               ('79001E', '780079'): 56, ('79001E', '79001C'): 57, ('790020', '79001E'): 58, ('790022', '790020'): 59,
               ('790022', '790064'): 60, ('790024', '790022'): 61, ('79005E', '790012'): 62, ('790060', '79005E'): 63,
               ('790062', '790060'): 64, ('790064', '78007B'): 65, ('790064', '790062'): 66, ('790064', '79007A'): 67,
               ('790068', '78007D'): 68, ('790068', '79007C'): 69, ('79006A', '790068'): 70, ('79006C', '79006A'): 71,
               ('79007A', '78001D'): 72, ('79007A', '79001C'): 73, ('79007C', '780063'): 74, ('79007C', '790062'): 75,
               ('79007C', '79007A'): 76, ('79007E', '780067'): 77, ('79007E', '79007C'): 78, ('790080', '79007E'): 79}
dragon_toll={('78000F', '2002'): 80, ('780013', '2008'): 81, ('780013', '2009'): 82, ('780019', '2007'): 83, ('78001B', '101001'): 84,
               ('78001D', '101007'): 85, ('780021', '2005'): 86, ('78005D', '2011'): 87, ('78005F', '2012'): 88, ('780067', '106006'): 89,
               ('780069', '106007'): 90, ('78007B', '102004'): 91, ('78007D', '102005'): 92, ('790014', '2002'): 93, ('79001A', '2008'): 94,
               ('79001A', '2009'): 95, ('79001C', '2007'): 96, ('79001E', '101001'): 97, ('790020', '101007'): 98, ('790024', '2005'): 99,
               ('790060', '2011'): 100, ('790062', '2012'): 101, ('790068', '102004'): 102, ('79006A', '106006'): 103, ('79006C', '106007'): 104,
               ('79007A', '101001'): 105, ('79007E', '102004'): 106, ('790080', '102005'): 107}

path_road_pair={'toll_dragon_15.csv':toll_dragon, 'dragon_dragon_15.csv':dragon_dragon, 'dragon_toll_15.csv':dragon_toll}
keys=['qidian', 'zhongdian', 'tian', 'xiaoshi', 'fenzhong','avg_sudu','cheliuliang'] # keys
months=[-1,31,29,31,30,31,30,31,31,30,31,30,31] # -1 represents a sentinel
hours=24    # 24 h
minutes=60  # 60 minutes

# data1=pd.read_csv('in1.5.csv',encoding='gb2312')
# print(data1.values[:10])
# print(data1.loc[(data1['日期']=='2021/6/1')])

import datetime

def empty_insert(data,date_time,low_month=6,high_month=9,day=1, hour=0,minute=0,index=0):
    # 本函数用于填充数据的，我们以周为周期，来填充我们的数据，我们设置最大跨度为2周
    date = datetime.datetime.strptime(date_time, "%Y/%m/%d") #

    date_day = datetime.datetime.strptime(date_time, "%Y/%m/%d") #

    while date.month<high_month: # 周期为7天的递增，取向上同周期内的速度数据
        date = date + datetime.timedelta(days=7)
        current_date=str(date.year)+'/'+str(date.month)+'/'+str(date.day)
        data_1 = data.loc[(data['tian'] == current_date) & (data['xiaoshi'] == hour) & (data['fenzhong'] == minute)]
        if not data_1.empty:
            return data_1.values[0,-2]

    while date_day.month<high_month: # 周期为1天的递增，取向上同周期内的速度数据
        date_day = date_day + datetime.timedelta(days=1)
        current_date=str(date_day.year)+'/'+str(date_day.month)+'/'+str(date_day.day)
        data_1 = data.loc[(data['tian'] == current_date) & (data['xiaoshi'] == hour) & (data['fenzhong'] == minute)]
        if not data_1.empty:
            return data_1.values[0,-2]

    data_1 = data.loc[(data['xiaoshi'] == hour)]
    if not data_1.empty:
        return np.mean(data_1.values[:,-2],dtype=np.float32)
    else:
        return np.mean([np.max(data.values[:,-2]),np.min(data.values[:,-2])])

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
    # used to store the DataFrame data of each station
    road_data_dict = {}
    for in_file_path in file_paths:
        road_data = pd.read_csv(filepath_or_buffer=in_file_path, encoding=encoding)
        road_data['qidian'] = road_data['qidian'].astype(str)
        road_data['zhongdian'] = road_data['zhongdian'].astype(str)
        for road_pair in path_road_pair[in_file_path].keys():
            print('the road pair name is : <%s, %s>, and the pair index is : %d' % (road_pair[0], road_pair[1],path_road_pair[in_file_path][(road_pair[0], road_pair[1])]))
            # read each station data
            road_data_dict[(road_pair[0],road_pair[1])]=road_data.loc[(road_data['qidian'] == road_pair[0])&(road_data['zhongdian'] == road_pair[1])]
            # use list to store each station data

    for month in range(beg_month,end_month):
        # to traverse the input months list
        for day in range(1, months[month]+1):
            # to traverse the input days of each month
            current_date=str(year)+'/'+str(month)+'/'+str(day)
            for hour in range(hours):
                for minute in range(0, minutes, 15):
                    for in_file_path in file_paths:
                        for road_pair in path_road_pair[in_file_path].keys(): # read data form the DataFrom list
                            data=road_data_dict[(road_pair[0],road_pair[1])]
                            data_1=data.loc[(data['tian'] == current_date) & (data['xiaoshi'] == hour) & (data['fenzhong'] == minute)]
                            if not data_1.empty:
                                print(path_road_pair[in_file_path][(road_pair[0],road_pair[1])], current_date, day, hour, minute, data_1.values[0,-2])
                                yield path_road_pair[in_file_path][(road_pair[0],road_pair[1])], current_date, day, hour, minute, data_1.values[0,-2]
                            else:
                                speed=empty_insert(data,date_time=current_date,low_month=beg_month,high_month=end_month, day=day, hour=hour,minute=minute)
                                print(path_road_pair[in_file_path][(road_pair[0], road_pair[1])], current_date, day,hour, minute, speed)
                                yield path_road_pair[in_file_path][(road_pair[0], road_pair[1])], current_date, day,hour, minute, speed

def data_combine(file_paths, out_path, beg_month=6,end_month=9,year=2021,encoding='utf-8'):
    '''
    :param file_paths: a list, contain original data sets
    :param out_path: write path, used to save the training set
    :return:
    '''
    file = open(out_path, 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(['node','date','day','hour','minute','speed'])
    for line in read_source(file_paths=file_paths, beg_month=beg_month,end_month=end_month, year=year, encoding=encoding):
        writer.writerow(line)
    file.close()

def adjacent(dict_1=None, dict_2=None, dict_3=None):
    file = open('adjacent.csv', 'w', encoding='utf-8')
    writer = csv.writer(file)
    for key in dict_1.keys():
        for key_2 in dict_2.keys():
            if key[1]==key_2[0] or key[1]==key_2[1]:
                writer.writerow([dict_1[(key[0],key[1])],dict_2[(key_2[0],key_2[1])]])
        for key_3 in dict_3.keys():
            if key[1]==key_3[0]:
                writer.writerow([dict_1[(key[0],key[1])],dict_3[(key_3[0],key_3[1])]])

    for key in dict_2.keys():
        for key_1 in dict_1.keys():
            if key[0]==key_1[1] or key[1]==key_1[1]:
                writer.writerow([dict_2[(key[0],key[1])],dict_1[(key_1[0],key_1[1])]])
        for key_2 in dict_2.keys():
            if key[1]==key_2[0] or key[0]==key_2[1]:
                if (key[0],key[1])!=(key_2[0],key_2[1]):
                    writer.writerow([dict_2[(key[0],key[1])],dict_2[(key_2[0],key_2[1])]])
        for key_3 in dict_3.keys():
            if key[1]==key_3[0] or key[0]==key_3[0]:
                writer.writerow([dict_2[(key[0],key[1])],dict_3[(key_3[0],key_3[1])]])

    for key in dict_3.keys():
        for key_2 in dict_2.keys():
            if key[0]==key_2[1] or key[0]==key_2[0]:
                writer.writerow([dict_3[(key[0],key[1])],dict_2[(key_2[0],key_2[1])]])
        for key_1 in dict_1.keys():
            if key[0]==key_1[1]:
                writer.writerow([dict_3[(key[0],key[1])],dict_1[(key_1[0],key_1[1])]])
    file.close()


if __name__=='__main__':
    print('hello')
    # data_combine(file_paths=['toll_dragon_15.csv', 'dragon_dragon_15.csv', 'dragon_toll_15.csv'], out_path='train.csv')
    # read_source(file_paths=['toll_dragon_15.csv', 'dragon_dragon_15.csv', 'dragon_toll_15.csv'], year=2021)
    # for line in read_source(file_paths=['in1.5.csv','in1.csv','in2.csv','in3.csv'],year=2021,encoding='gb2312'):
    #     print(line)
    #
    # data = pd.read_csv('train.csv', encoding='utf-8').values
    # print(data.shape)

    # file = open('in_flow_train.csv', 'w', encoding='utf-8')
    # writer = csv.writer(file)
    # writer.writerow(['station','date','hour','minute','flow'])
    # for line in data:
    #     writer.writerow([str(line[0]),str(line[1]),int(line[2]),int(line[3]),int(line[4])])
    # file.close()

    adjacent(toll_dragon,dragon_dragon,dragon_toll)

    print('finished')

    # road_pairs={}
    # index=0
    # file_paths = ['toll_dragon_15.csv', 'dragon_dragon_15.csv', 'dragon_toll_15.csv']
    # for file_path in file_paths:
    #     data=pd.read_csv(file_path,encoding='utf-8',usecols=['qidian','zhongdian']).values
    #     for line in data:
    #         if (str(line[0]),str(line[1])) not in road_pairs:
    #             # print((str(line[0]),str(line[1])))
    #             road_pairs[(str(line[0]),str(line[1]))]=index
    #             index+=1
    #         else:continue
    #
    # print(road_pairs)