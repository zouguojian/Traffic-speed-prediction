# -- coding: utf-8 --
import pandas as pd
import numpy as np
print(np.timedelta64(2,"D"))

file='/Users/guojianzou/Traffic-speed-prediction/MT-STGIN/data/speed/train_15.csv'
df = pd.read_csv(file)

df['datatime'] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h') + pd.to_timedelta(
    df.minute,
    unit='m')

per = pd.Period('2017-12-31 22:00', '4H')
print(per.day_of_week)

time_ind = (df['datatime'].values - df['datatime'].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
# k=0
# for line in time_ind:
#     if line==0.0:
#         k+=1
#         print(k)
time_in_day = np.tile(time_ind, [1, 108, 1]).transpose((2, 1, 0))
print(time_ind.shape)
print(time_in_day.shape)