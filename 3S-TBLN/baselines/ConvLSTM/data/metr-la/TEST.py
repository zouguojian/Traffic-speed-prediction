# -- coding: utf-8 --
import pandas as pd
import numpy as np
#
# data = pd.read_hdf('/Users/guojianzou/Traffic-speed-prediction/3S-TAEN/data/metr-la/metr-la.h5')
# print(data.shape)


import h5py

#h5 file path
filename = '/Users/guojianzou/Traffic-speed-prediction/3S-TAEN/data/metr-la/metr-la.h5'

#read h5 file
dataset = h5py.File(filename, 'r')

# data = np.load('/Users/guojianzou/Traffic-speed-prediction/3S-TAEN/data/metr-la/METR-LA.npz')
# print(data['data'].shape)

import pickle

F=open(r'/Users/guojianzou/Traffic-speed-prediction/3S-TAEN/data/metr-la/adj_mx.pkl','rb')

content=pickle.load(F,encoding='iso-8859-1')
print(len(content))
print(content[0])
print(content[1])
print(content[2])

csv_data_a = pd.read_csv('/Users/guojianzou/Traffic-speed-prediction/3S-TAEN/data/metr-la/metr-la.csv')
print(list(csv_data_a.keys()))


print('2017/454/3'.replace('/','-'))

