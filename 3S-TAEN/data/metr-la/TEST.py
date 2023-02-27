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

#print the first unknown key in the h5 file

# group=dataset['df'] #returns df
# for key in group.keys():
#     print(group[key].value)

#save the h5 file to csv using the first key df
'''
with pd.HDFStore(filename, 'r') as d:
    df = d.get('df')
    df.to_csv('metr-la.csv')
'''

# data = np.load('/Users/guojianzou/Traffic-speed-prediction/3S-TAEN/data/metr-la/METR-LA.npz')
# print(data['data'].shape)

