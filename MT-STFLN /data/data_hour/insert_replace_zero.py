# -- coding: utf-8 --
import pandas as pd
import numpy as np
import csv

source_data=r'train.csv'
insert_data=r'train_insert.csv'

data=pd.read_csv(insert_data,encoding='utf-8')

site_index=data.values[:49,:2]  # 站点对，即道路
dict_index={(site[0],site[1]):None for site in site_index}
print(dict_index)

import matplotlib.pyplot as plt
def describe(label):
    '''
    :param label:
    :param predict:
    :param prediction_size:
    :return:
    '''
    plt.figure()
    # Label is observed value,Blue
    plt.plot(label, 'b', label=u'actual value')
    # Predict is predicted value，Red
    # plt.plot(predict[0:], 'r', label=u'predicted value')
    # use the legend
    plt.legend()
    # plt.xlabel("time(hours)", fontsize=17)
    # plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
    # plt.title("the prediction of pm$_{2.5}", fontsize=17)
    plt.show()

'''
1.如果你的数据增长速率越来越快，可以选择 method='quadratic'二次插值。
2.如果数据集呈现出累计分布的样子，推荐选择 method='pchip'。
3.如果需要填补缺省值，以平滑绘图为目标，推荐选择 method='akima'。
4.默认为 linear
'''

for (site1, site2) in dict_index.keys():
    data1 = data[(data['in_id'] == site1) & (data['out_id'] == site2)]  # 获取制定道路数据
    describe(data1.values[:,6])

def insert():
    for (site1,site2) in dict_index.keys():
        data1 = data[(data['in_id'] == site1) & (data['out_id'] == site2)] # 获取制定道路数据
        data1['speed'].replace(0, np.nan, inplace=True) # 替换掉数据为0的道路
        data1.interpolate(method='linear', inplace=True) # 数据填充
        data1=data1.iloc[::-1]
        data1.interpolate(method='linear', inplace=True)  # 数据填充
        data1 = data1.iloc[::-1]
        for line in data1.values:
            print(list(line))
        if (site1,site2) in dict_index:
            print('writing')
            dict_index[(site1,site2)]=data1.values

    new_key=data.keys()
    file = open(insert_data, 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(new_key)

    for i in range(2832): # 每个道路数据的个数
        for (site1,site2) in dict_index.keys():
            writer.writerow(list(dict_index[(site1,site2)][i]))
    file.close()

    print('finish')


