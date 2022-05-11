# -- coding: utf-8 --

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


def show(a, b, x1, x2):
    plt.figure()
    plt.plot(x1,a)
    plt.plot(x2, b,color='red')
    plt.show()


# data=pd.read_csv('train_p.csv',encoding='utf-8')
# keys=['month', 'day', 'hour', 'min','AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
# file = open('train_p_', 'w', encoding='utf-8')
# writer = csv.writer(file)
#
# writer.writerow(keys)
#
# print(float('nan'))
#
# for line in data.values:
#     line = list(line)
#     for min in range(0,60,15):
#         if min==0:
#             writer.writerow(line[:3]+[min]+line[3:])
#         else:
#             writer.writerow(line[:3]+[min]+ [float('nan') for _ in range(len(line[3:]))])
# file.close()

data=pd.read_csv('train_p_.csv',encoding='utf-8')
data.interpolate(method='pchip',axis=4,inplace=True)

print(data.values)