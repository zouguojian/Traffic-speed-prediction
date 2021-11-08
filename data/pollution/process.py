# -- coding: utf-8 --

# -- coding: utf-8 --
import os
import pandas as pd
import numpy as np
import csv
import datetime
#
# dd = '201903173'
# dd = datetime.datetime.strptime(dd, "%Y%m%d%H")
# print(dd,type(dd))


def pollution():
    rootdir = '/Users/guojianzou/Downloads/站点_20200101-20201231'

    # '1146A'
    # 3271A

    keys=['Time', 'AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
    file = open('pollution_2020.csv', 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(keys)

    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    list.sort()
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isfile(path):
              print(path)
              data=pd.read_csv(path,usecols=['date','hour','type','1146A']).values
              # data=np.transpose(data,[1,0])
              print(data.shape)

              for i in range(0, data.shape[0], 15):
                  date=str(data[i][0]) + str(data[i][1])
                  date = datetime.datetime.strptime(date, "%Y%m%d%H")
                  print(date)
                  feaure=data[i:i+15,3]
                  # print(feaure)

                  writer.writerow([date]+feaure.tolist())
    file.close()


import matplotlib.pyplot as plt

data1=pd.read_csv('train_p.csv', encoding='utf-8').values[:,8]
# data2=pd.read_csv('pollution_2021.csv', encoding='utf-8').values[:500,1]

plt.figure()
# Label is observed value,Blue
plt.plot(data1, 'b*:', label=u'actual value')
# Predict is predicted value，Red
# plt.plot(np.concatenate((data1,data2),axis=0), 'r*:', label=u'predicted value')
# use the legend
# plt.legend()
plt.xlabel("time(hours)", fontsize=17)
plt.ylabel("pm$_{10}$ (ug/m$^3$)", fontsize=17)
plt.title("the prediction of pm$_{10}$", fontsize=17)
plt.show()

def combine_pollution():
    data1=pd.read_csv('pollution_2020.csv', encoding='utf-8').values
    data2=pd.read_csv('pollution_2021.csv', encoding='utf-8').values

    keys=['Time', 'AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
    file = open('pollution_train.csv', 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(keys)

    for line in data1[4355:]:
        writer.writerow(line)
    for line in data2[:5792]:
        writer.writerow(line)

    file.close()
# combine_pollution()



# 线性插值
def liner_insert(d1,key_0,key,x,y):
    d = pd.DataFrame()
    d['date'] = x
    d['val'] = y
    d['date'] = pd.to_datetime(d['date'])
    helper = pd.DataFrame({'date': pd.date_range(d['date'].min(), d['date'].max(), freq='H')})

    d = pd.merge(d, helper, on='date', how='outer').sort_values('date')
    d['val'] = d['val'].interpolate(method='linear')
    print(d.values.shape)
    print(d.values)
    if key_0 not in d1:
        d1[key_0]=d.values[:,0]
    d1[key]=d.values[:,1]
    # print(d['val'].values.shape)
    # helper = pd.DataFrame({'date': pd.date_range(d['date'].min(), d['date'].max(), freq='H')})


def train_():
    data=pd.read_csv('pollution_train.csv', encoding='utf-8')
    keys=['Time', 'AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
    d1 = pd.DataFrame()

    for key in keys[1:]:
        x,y=[],[]
        for i in range(data.values.shape[0]):
            if np.isnan(data[key].values[i]):continue
            else:
                x.append(data.values[:,0][i])
                y.append(data[key].values[i])

        liner_insert(d1,keys[0],key,x,y)

    file = open('train_p.csv', 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(keys)
    for i in range(d1.values.shape[0]):
        writer.writerow(list(d1.values[i]))
    file.close()

# train_()


def ymdh():
    data=pd.read_csv('train_p.csv', encoding='utf-8')
    keys=['month', 'day', 'hour', 'AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
    file = open('train_pp.csv', 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(keys)

    for line in data.values:
        m,d,h=str(line[0])[5:7],str(line[0])[8:10],str(line[0])[11:13]
        print(m,d,h)
        writer.writerow([float(m), float(d), float(h)]+list(line[1:]))
    file.close()

# ymdh()