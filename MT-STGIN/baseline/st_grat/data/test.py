# -- coding: utf-8 --

import numpy as np
import pandas as pd

# da=pd.read_csv('/Users/guojianzou/Traffic-speed-prediction/STGIN/baseline/st_grat/data/PEMSD4/PEMSD4.dyna')
# print(da.keys())
# print(da.values[-1])

# data = np.load('/Users/guojianzou/Traffic-speed-prediction/STGIN/baseline/st_grat/data/pems04.npz')['data']
# print(data.shape)

data = np.load('/Users/guojianzou/Traffic-speed-prediction/STGIN/baseline/st_grat/data/we/train.npz',allow_pickle=True)['data']
print(data.shape)