# -- coding: utf-8 --
import pandas as pd
import h5py
import csv
from datetime import datetime
import numpy as np


""" 数据从h5解析道csv格式 """
'''
# h5 file path
filename = 'metr-la.h5'

#read h5 file
dataset = h5py.File(filename, 'r')

#print the first unknown key in the h5 file
print(list(dataset.keys()))

with pd.HDFStore(filename, 'r') as d:
    df = d.get('df')
    print(df.shape)
    print(df)
    df.to_csv('metr-la.csv')
'''


# ---------------------------------------------------------------------------------------------
""" 数据转化到我们需要的固定格式 """
keys =['node', 'date', 'day', 'hour', 'minute', 'speed']
def convert_a_to_b(file_a=None, file_b=None, site=207, keys=None):
    file = open(file_b, 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(keys)

    csv_data_a = pd.read_csv(file_a).values
    csv_data_a = csv_data_a.T # 转置
    new_shape = csv_data_a.shape
    print(new_shape)

    for i in range(round(new_shape[1] * (3 / 4))):
        time = datetime.strptime(str(csv_data_a[0, i]), '%Y-%m-%d %H:%M:%S')
        for index in range(1, new_shape[0]):
            node = index-1
            date = str(time)[:10]
            day = time.day
            hour = time.hour
            minute = time.minute
            speed = csv_data_a[index, i]
            writer.writerow([node, date, day, hour, minute, speed])
    file.close()
    return

'''
convert_a_to_b(file_a='metr-la.csv',
               file_b='train.csv',
               site=207,
               keys=keys)
print('finish')
'''


# ---------------------------------------------------------------------------------------------

""" 邻接矩阵生成 """

import pickle

'''
def adjacent(file_a=None, file_b=None):
    keys = ['src_FID', 'nbr_FID', 'weight']
    file = open(file_b, 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(keys)

    F = open(file_a, 'rb')
    content = pickle.load(F, encoding='iso-8859-1')
    print(len(content))
    print(content[0],len(content[0]))
    print(content[1],len(content[1]))
    print(content[2])
    adjacent_m = content[2]
    shape = adjacent_m.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            if adjacent_m[i,j]>0:
                writer.writerow([i, j, adjacent_m[i,j]])
            else:continue
    file.close()
    return

adjacent(file_a='adj_mx.pkl',
         file_b='adjacent.csv')
print('finish')

F=open(r'adj_mx.pkl','rb')
content=pickle.load(F,encoding='iso-8859-1')
print(len(content))
print(content[0])
print(content[1])
print(content[2])
'''

# ---------------------------------------------------------------------------------------------

""" 数据填充, 直接存储, 不需要再次转换操作 """
def week_insert(data=None, index=1, granularity=5, left_limit=0, right_limit=1000000):
    # 左边填充到边界的话, 尝试右边, 否则返回False
    l_index=r_index=index
    while l_index > left_limit:
        if data[l_index] != 0:
            return data[l_index]
        else:
            l_index -= (60 // granularity * 24 * 7)

    while r_index < right_limit:
        if data[r_index] != 0:
            return data[r_index]
        else:
            r_index += (60 // granularity * 24 * 7)
    return data[index-1]

def fill_data(file_a=None, file_b=None, site=208, keys=None):
    file = open(file_b, 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(keys)
    data_n = pd.read_csv(file_a).values

    shape = data_n.shape
    for index in range(1, shape[1]):
        for row in range(shape[0]):
            if data_n[row,index] == 0:
                data_n[row, index] = week_insert(data=data_n[:,index],index=row,granularity=5,left_limit=0, right_limit=shape[0])
        print('the %d-th col has been scanned'% (index))

    # csv_data_a = pd.read_csv(file_a).values
    csv_data_a = data_n
    csv_data_a = csv_data_a.T # 转置
    new_shape = csv_data_a.shape
    print(new_shape)

    for i in range(new_shape[1]):
        time = datetime.strptime(str(csv_data_a[0, i]), '%Y-%m-%d %H:%M:%S')
        for index in range(1, new_shape[0]):
            node = index-1
            date = str(time)[:10]
            day = time.day
            hour = time.hour
            minute = time.minute
            speed = csv_data_a[index, i]
            writer.writerow([node, date, day, hour, minute, speed])
    file.close()
    return

# fill_data(file_a='/Users/guojianzou/Traffic-speed-prediction/3S-TAEN/data/metr-la/metr-la.csv',
#                file_b='/Users/guojianzou/Traffic-speed-prediction/3S-TAEN/data/metr-la/train_5.csv',
#                site=208,
#                keys=keys)
# print('finish')


import matplotlib.pyplot as plt
def describe(label):
    '''
    :param label:
    :param predict:
    :param prediction_size:
    :return:
    '''
    plt.figure()
    plt.plot(label, 'b', label=u'actual value')
    plt.legend()
    plt.show()

# data_n = pd.read_csv('/Users/guojianzou/Traffic-speed-prediction/3S-TAEN/data/metr-la/metr-la.csv').values
# shape = data_n.shape
# for index in range(1, shape[1]):
#     for row in range(shape[0]):
#         if data_n[row, index] == 0:
#             data_n[row, index] = week_insert(data=data_n[:, index], index=row, granularity=5, left_limit=0,
#                                              right_limit=shape[0])
#     print('the %d-th col has been scanned' % (index))
#     describe(label=list(data_n[:,index]))


data = np.load("test.npz",allow_pickle=True)['y']
print(np.min(data[:,0]))