# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from models.inits import *
from matplotlib.ticker import MaxNLocator
import seaborn as sns

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 11.,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 11,
}


DCRNN_LA = np.load('dcrnn_predictions-la.npz')['prediction'].transpose([1,2,0])[19:]
LABEL_LA = np.load('dcrnn_predictions-la.npz')['truth'].transpose([1,2,0])[19:]
DCRNN_YC = np.load('dcrnn_predictions-yc.npz')['prediction'].transpose([1,2,0])[19:]
LABEL_YC = np.load('dcrnn_predictions-yc.npz')['truth'].transpose([1,2,0])[19:]
AGCRN_LA = np.load('AGCRN-METR_LA.npz')['prediction'].squeeze(axis=-1).transpose([0,2,1])[19:]
AGCRN_YC = np.load('AGCRN-YINCHUAN.npz')['prediction'].squeeze(axis=-1).transpose([0,2,1])[19:]
ASTGCN_LA = np.load('ASTGCN-METR-LA.npz')['prediction'][19:]
ASTGCN_YC = np.load('ASTGCN-YINCHUAN.npz')['prediction'][19:]
Graph_WaveNet_LA = np.load('Graph-METR.npz')['prediction'][19:]
Graph_WaveNet_YC = np.load('Graph-YINCHUAN.npz')['prediction'][19:]
GMAN_LA = np.load('GMAN-METR.npz', allow_pickle=True)['prediction']
GMAN_YC = np.load('GMAN-YINCHUAN.npz', allow_pickle=True)['prediction']
ST_GRAT_LA = np.load('ST-GRAT-METR-LA.npz', allow_pickle=True)['prediction'].transpose([0,2,1])[19:]
ST_GRAT_YC = np.load('ST-GRAT-YINCHUAN.npz', allow_pickle=True)['prediction'].transpose([0,2,1])[19:]
MTGNN_LA = np.load('MTGNN-METR-LA.npz', allow_pickle=True)['prediction'][19:]
MTGNN_YC = np.load('MTGNN-YINCHUAN.npz', allow_pickle=True)['prediction'][19:]
TBLN_LA = np.load('TBLN-METR-LA.npz', allow_pickle=True)['prediction']
TBLN_YC = np.load('TBLN-YINCHUAN.npz', allow_pickle=True)['prediction']

STGIN_YC = np.load('STGIN-YINCHUAN.npz', allow_pickle=True)['prediction']

print(DCRNN_LA.shape)
print(DCRNN_YC.shape)
print(AGCRN_LA.shape)
print(AGCRN_YC.shape)
print(ASTGCN_LA.shape)
print(ASTGCN_YC.shape)
print(Graph_WaveNet_LA.shape)
print(Graph_WaveNet_YC.shape)
print(GMAN_LA.shape)
print(GMAN_YC.shape)
print(ST_GRAT_LA.shape)
print(ST_GRAT_YC.shape)
print(MTGNN_LA.shape)
print(MTGNN_YC.shape)
print(TBLN_LA.shape)
print(TBLN_YC.shape)



# '''
# 用于展示YINCHUAN中600个测试样本的拟合程度
plt.figure()
mean = 80.13087428380071
std = 30.2782227196378

plt.subplot(3, 1, 1)
road_index = 22
total=1200
plt.plot(np.concatenate([list(LABEL_YC[sample_index, road_index] * std + mean) for sample_index in range(12, total, 12)],axis=-1), color='black', linestyle='-',
         linewidth=0.7, label='Observed')
plt.plot(np.concatenate([list(DCRNN_YC[sample_index, road_index] * std + mean) for sample_index in range(12, total, 12)], axis=-1), color='red', linestyle='-',
         linewidth=0.5, label='DCRNN')
plt.plot(np.concatenate([list(AGCRN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1),  color='blue', linestyle='-', linewidth=0.5,
         label='AGCRN')
plt.plot(np.concatenate([list(ASTGCN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='orange', linestyle='-', linewidth=0.5,
         label='ASTGCN')
plt.plot(np.concatenate([list(Graph_WaveNet_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#0cdc73', linestyle='-',
         linewidth=0.5, label='Graph-WaveNet')
plt.plot(np.concatenate([list(GMAN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#f504c9', linestyle='-', linewidth=0.5,
         label='GMAN')
plt.plot(np.concatenate([list(ST_GRAT_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#d0c101', linestyle='-',
         linewidth=0.5, label='ST-GRAT')
plt.plot(np.concatenate([list(MTGNN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#ff5b00', linestyle='-', linewidth=0.5,
         label='MTGNN')
plt.plot(np.concatenate([list(TBLN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
         label='3S-TBLN')
# plt.plot(np.concatenate([list(STGIN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
#          label='STGIN')
plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic speed (km/h)', font2)
# plt.xlabel('Target time steps', font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 2)
road_index = 66
plt.plot(np.concatenate([list(LABEL_YC[sample_index, road_index] * std + mean) for sample_index in range(12, total, 12)],axis=-1), color='black', linestyle='-',
         linewidth=0.7, label='Observed')
plt.plot(np.concatenate([list(DCRNN_YC[sample_index, road_index] * std + mean) for sample_index in range(12, total, 12)], axis=-1), color='red', linestyle='-',
         linewidth=0.5, label='DCRNN')
plt.plot(np.concatenate([list(AGCRN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1),  color='blue', linestyle='-', linewidth=0.5,
         label='AGCRN')
plt.plot(np.concatenate([list(ASTGCN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='orange', linestyle='-', linewidth=0.5,
         label='ASTGCN')
plt.plot(np.concatenate([list(Graph_WaveNet_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#0cdc73', linestyle='-',
         linewidth=0.5, label='Graph-WaveNet')
plt.plot(np.concatenate([list(GMAN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#f504c9', linestyle='-', linewidth=0.5,
         label='GMAN')
plt.plot(np.concatenate([list(ST_GRAT_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#d0c101', linestyle='-',
         linewidth=0.5, label='ST-GRAT')
plt.plot(np.concatenate([list(MTGNN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#ff5b00', linestyle='-', linewidth=0.5,
         label='MTGNN')
plt.plot(np.concatenate([list(TBLN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
         label='3S-TBLN')
# plt.plot(np.concatenate([list(STGIN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
#          label='STGIN')
# plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic speed (km/h)', font2)
# plt.xlabel('Target time steps', font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 3)
road_index = 75
plt.plot(np.concatenate([list(LABEL_YC[sample_index, road_index] * std + mean) for sample_index in range(12, total, 12)],axis=-1), color='black', linestyle='-',
         linewidth=0.7, label='Observed')
plt.plot(np.concatenate([list(DCRNN_YC[sample_index, road_index] * std + mean) for sample_index in range(12, total, 12)], axis=-1), color='red', linestyle='-',
         linewidth=0.5, label='DCRNN')
plt.plot(np.concatenate([list(AGCRN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1),  color='blue', linestyle='-', linewidth=0.5,
         label='AGCRN')
plt.plot(np.concatenate([list(ASTGCN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='orange', linestyle='-', linewidth=0.5,
         label='ASTGCN')
plt.plot(np.concatenate([list(Graph_WaveNet_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#0cdc73', linestyle='-',
         linewidth=0.5, label='Graph-WaveNet')
plt.plot(np.concatenate([list(GMAN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#f504c9', linestyle='-', linewidth=0.5,
         label='GMAN')
plt.plot(np.concatenate([list(ST_GRAT_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#d0c101', linestyle='-',
         linewidth=0.5, label='ST-GRAT')
plt.plot(np.concatenate([list(MTGNN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#ff5b00', linestyle='-', linewidth=0.5,
         label='MTGNN')
plt.plot(np.concatenate([list(TBLN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
         label='3S-TBLN')
# plt.plot(np.concatenate([list(STGIN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
#          label='STGIN')
# plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic speed (km/h)', font2)
# plt.xlabel('Target time steps', font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
# '''


# YC实际的预测过程展示，样例
# '''
mean = 80.13087428380071
std = 30.2782227196378

plt.subplot(3, 1, 1)
road_index = 22
sample_index =600
plt.plot(range(1, 14, 1), np.concatenate(
    [LABEL_YC[sample_index - 12, road_index] * std + mean, LABEL_YC[sample_index, road_index, 0:1] * std + mean],
    axis=-1), marker='P', color='black', linestyle='--', linewidth=1)
plt.plot(range(13, 25, 1), LABEL_YC[sample_index, road_index] * std + mean, marker='P', color='black', linestyle='--',
         linewidth=1, label='Observed')
plt.plot(range(13, 25, 1), DCRNN_YC[sample_index, road_index] * std + mean, marker='P', color='red', linestyle='-',
         linewidth=1, label='DCRNN')
plt.plot(range(13, 25, 1), AGCRN_YC[sample_index, road_index], marker='h', color='blue', linestyle='-', linewidth=1,
         label='AGCRN')
plt.plot(range(13, 25, 1), ASTGCN_YC[sample_index, road_index], marker='o', color='orange', linestyle='-', linewidth=1,
         label='ASTGCN')
plt.plot(range(13, 25, 1), Graph_WaveNet_YC[sample_index, road_index], marker='s', color='#0cdc73', linestyle='-',
         linewidth=1, label='Graph-WaveNet')
plt.plot(range(13, 25, 1), GMAN_YC[sample_index, road_index], marker='p', color='#f504c9', linestyle='-', linewidth=1,
         label='GMAN')
plt.plot(range(13, 25, 1), ST_GRAT_YC[sample_index, road_index], marker='^', color='#d0c101', linestyle='-',
         linewidth=1, label='ST-GRAT')
plt.plot(range(13, 25, 1), MTGNN_YC[sample_index, road_index], marker='d', color='#ff5b00', linestyle='-', linewidth=1,
         label='MTGNN')
plt.plot(range(13, 25, 1), TBLN_YC[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
         label='3S-TBLN')
# plt.plot(range(13, 25, 1), STGIN_YC[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
#          label='STGIN')
plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic speed (km/h)', font2)
# plt.xlabel('Number of samples',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 2)
road_index = 66
sample_index =600
plt.plot(range(1, 14, 1), np.concatenate(
    [LABEL_YC[sample_index - 12, road_index] * std + mean, LABEL_YC[sample_index, road_index, 0:1] * std + mean],
    axis=-1), marker='P', color='black', linestyle='--', linewidth=1)
plt.plot(range(13, 25, 1), LABEL_YC[sample_index, road_index] * std + mean, marker='P', color='black', linestyle='--',
         linewidth=1, label='Observed')
plt.plot(range(13, 25, 1), DCRNN_YC[sample_index, road_index] * std + mean, marker='P', color='red', linestyle='-',
         linewidth=1, label='DCRNN')
plt.plot(range(13, 25, 1), AGCRN_YC[sample_index, road_index], marker='h', color='blue', linestyle='-', linewidth=1,
         label='AGCRN')
plt.plot(range(13, 25, 1), ASTGCN_YC[sample_index, road_index], marker='o', color='orange', linestyle='-', linewidth=1,
         label='ASTGCN')
plt.plot(range(13, 25, 1), Graph_WaveNet_YC[sample_index, road_index], marker='s', color='#0cdc73', linestyle='-',
         linewidth=1, label='Graph-WaveNet')
plt.plot(range(13, 25, 1), GMAN_YC[sample_index, road_index], marker='p', color='#f504c9', linestyle='-', linewidth=1,
         label='GMAN')
plt.plot(range(13, 25, 1), ST_GRAT_YC[sample_index, road_index], marker='^', color='#d0c101', linestyle='-',
         linewidth=1, label='ST-GRAT')
plt.plot(range(13, 25, 1), MTGNN_YC[sample_index, road_index], marker='d', color='#ff5b00', linestyle='-', linewidth=1,
         label='MTGNN')
plt.plot(range(13, 25, 1), TBLN_YC[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
         label='3S-TBLN')
# plt.plot(range(13, 25, 1), STGIN_YC[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
#          label='STGIN')
# plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic speed (km/h)', font2)
# plt.xlabel('Number of samples',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 3)
road_index = 75
sample_index =600
plt.plot(range(1, 14, 1), np.concatenate(
    [LABEL_YC[sample_index - 12, road_index] * std + mean, LABEL_YC[sample_index, road_index, 0:1] * std + mean],
    axis=-1), marker='P', color='black', linestyle='--', linewidth=1)
plt.plot(range(13, 25, 1), LABEL_YC[sample_index, road_index] * std + mean, marker='P', color='black', linestyle='--',
         linewidth=1, label='Observed')
plt.plot(range(13, 25, 1), DCRNN_YC[sample_index, road_index] * std + mean, marker='P', color='red', linestyle='-',
         linewidth=1, label='DCRNN')
plt.plot(range(13, 25, 1), AGCRN_YC[sample_index, road_index], marker='h', color='blue', linestyle='-', linewidth=1,
         label='AGCRN')
plt.plot(range(13, 25, 1), ASTGCN_YC[sample_index, road_index], marker='o', color='orange', linestyle='-', linewidth=1,
         label='ASTGCN')
plt.plot(range(13, 25, 1), Graph_WaveNet_YC[sample_index, road_index], marker='s', color='#0cdc73', linestyle='-',
         linewidth=1, label='Graph-WaveNet')
plt.plot(range(13, 25, 1), GMAN_YC[sample_index, road_index], marker='p', color='#f504c9', linestyle='-', linewidth=1,
         label='GMAN')
plt.plot(range(13, 25, 1), ST_GRAT_YC[sample_index, road_index], marker='^', color='#d0c101', linestyle='-',
         linewidth=1, label='ST-GRAT')
plt.plot(range(13, 25, 1), MTGNN_YC[sample_index, road_index], marker='d', color='#ff5b00', linestyle='-', linewidth=1,
         label='MTGNN')
plt.plot(range(13, 25, 1), TBLN_YC[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
         label='3S-TBLN')
# plt.plot(range(13, 25, 1), STGIN_YC[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
#          label='STGIN')
# plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic speed (km/h)', font2)
# plt.xlabel('Number of samples',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
# '''


# '''
# 用于展示METR中600个测试样本的拟合程度
plt.figure()
mean = 54.40592829587626
std = 19.493739270573094

plt.subplot(3, 1, 1)
road_index = 71
plt.plot(np.concatenate([list(LABEL_LA[sample_index, road_index] * std + mean) for sample_index in range(12, 1200, 12)],axis=-1), color='black', linestyle='-',
         linewidth=0.7, label='Observed')
plt.plot(np.concatenate([list(DCRNN_LA[sample_index, road_index] * std + mean) for sample_index in range(12, 1200, 12)], axis=-1), color='red', linestyle='-',
         linewidth=0.5, label='DCRNN')
plt.plot(np.concatenate([list(AGCRN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1),  color='blue', linestyle='-', linewidth=0.5,
         label='AGCRN')
plt.plot(np.concatenate([list(ASTGCN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='orange', linestyle='-', linewidth=0.5,
         label='ASTGCN')
plt.plot(np.concatenate([list(Graph_WaveNet_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#0cdc73', linestyle='-',
         linewidth=0.5, label='Graph-WaveNet')
plt.plot(np.concatenate([list(GMAN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#f504c9', linestyle='-', linewidth=0.5,
         label='GMAN')
plt.plot(np.concatenate([list(ST_GRAT_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#d0c101', linestyle='-',
         linewidth=0.5, label='ST-GRAT')
plt.plot(np.concatenate([list(MTGNN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#ff5b00', linestyle='-', linewidth=0.5,
         label='MTGNN')
plt.plot(np.concatenate([list(TBLN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
         label='3S-TBLN')
plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic speed (mile/h)', font2)
# plt.xlabel('Target time steps', font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 2)
road_index = 155
plt.plot(np.concatenate([list(LABEL_LA[sample_index, road_index] * std + mean) for sample_index in range(12, 1200, 12)],axis=-1), color='black', linestyle='-',
         linewidth=0.7, label='Observed')
plt.plot(np.concatenate([list(DCRNN_LA[sample_index, road_index] * std + mean) for sample_index in range(12, 1200, 12)], axis=-1), color='red', linestyle='-',
         linewidth=0.5, label='DCRNN')
plt.plot(np.concatenate([list(AGCRN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1),  color='blue', linestyle='-', linewidth=0.5,
         label='AGCRN')
plt.plot(np.concatenate([list(ASTGCN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='orange', linestyle='-', linewidth=0.5,
         label='ASTGCN')
plt.plot(np.concatenate([list(Graph_WaveNet_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#0cdc73', linestyle='-',
         linewidth=0.5, label='Graph-WaveNet')
plt.plot(np.concatenate([list(GMAN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#f504c9', linestyle='-', linewidth=0.5,
         label='GMAN')
plt.plot(np.concatenate([list(ST_GRAT_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#d0c101', linestyle='-',
         linewidth=0.5, label='ST-GRAT')
plt.plot(np.concatenate([list(MTGNN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#ff5b00', linestyle='-', linewidth=0.5,
         label='MTGNN')
plt.plot(np.concatenate([list(TBLN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
         label='3S-TBLN')
# plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic speed (mile/h)', font2)
# plt.xlabel('Target time steps', font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 3)
road_index = 187
plt.plot(np.concatenate([list(LABEL_LA[sample_index, road_index] * std + mean) for sample_index in range(12, 1200, 12)],axis=-1), color='black', linestyle='-',
         linewidth=0.7, label='Observed')
plt.plot(np.concatenate([list(DCRNN_LA[sample_index, road_index] * std + mean) for sample_index in range(12, 1200, 12)], axis=-1), color='red', linestyle='-',
         linewidth=0.5, label='DCRNN')
plt.plot(np.concatenate([list(AGCRN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1),  color='blue', linestyle='-', linewidth=0.5,
         label='AGCRN')
plt.plot(np.concatenate([list(ASTGCN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='orange', linestyle='-', linewidth=0.5,
         label='ASTGCN')
plt.plot(np.concatenate([list(Graph_WaveNet_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#0cdc73', linestyle='-',
         linewidth=0.5, label='Graph-WaveNet')
plt.plot(np.concatenate([list(GMAN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#f504c9', linestyle='-', linewidth=0.5,
         label='GMAN')
plt.plot(np.concatenate([list(ST_GRAT_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#d0c101', linestyle='-',
         linewidth=0.5, label='ST-GRAT')
plt.plot(np.concatenate([list(MTGNN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#ff5b00', linestyle='-', linewidth=0.5,
         label='MTGNN')
plt.plot(np.concatenate([list(TBLN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
         label='3S-TBLN')
# plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic speed (mile/h)', font2)
# plt.xlabel('Target time steps', font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
# '''



# METR实际的预测过程展示，样例
# '''
mean = 54.40592829587626
std = 19.493739270573094

plt.subplot(3, 1, 1)
road_index = 71
sample_index =400
plt.plot(range(1, 14, 1), np.concatenate(
    [LABEL_LA[sample_index - 12, road_index] * std + mean, LABEL_LA[sample_index, road_index, 0:1] * std + mean],
    axis=-1), marker='P', color='black', linestyle='--', linewidth=1)
plt.plot(range(13, 25, 1), LABEL_LA[sample_index, road_index] * std + mean, marker='P', color='black', linestyle='--',
         linewidth=1, label='Observed')
plt.plot(range(13, 25, 1), DCRNN_LA[sample_index, road_index] * std + mean, marker='P', color='red', linestyle='-',
         linewidth=1, label='DCRNN')
plt.plot(range(13, 25, 1), AGCRN_LA[sample_index, road_index], marker='h', color='blue', linestyle='-', linewidth=1,
         label='AGCRN')
plt.plot(range(13, 25, 1), ASTGCN_LA[sample_index, road_index], marker='o', color='orange', linestyle='-', linewidth=1,
         label='ASTGCN')
plt.plot(range(13, 25, 1), Graph_WaveNet_LA[sample_index, road_index], marker='s', color='#0cdc73', linestyle='-',
         linewidth=1, label='Graph-WaveNet')
plt.plot(range(13, 25, 1), GMAN_LA[sample_index, road_index], marker='p', color='#f504c9', linestyle='-', linewidth=1,
         label='GMAN')
plt.plot(range(13, 25, 1), ST_GRAT_LA[sample_index, road_index], marker='^', color='#d0c101', linestyle='-',
         linewidth=1, label='ST-GRAT')
plt.plot(range(13, 25, 1), MTGNN_LA[sample_index, road_index], marker='d', color='#ff5b00', linestyle='-', linewidth=1,
         label='MTGNN')
plt.plot(range(13, 25, 1), TBLN_LA[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
         label='3S-TBLN')
plt.legend(loc='upper left', prop=font1)
plt.grid(axis='y')
plt.ylabel('Traffic speed (mile/h)', font2)
# plt.xlabel('Number of samples',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 2)
road_index = 155
plt.plot(range(1, 14, 1), np.concatenate(
    [LABEL_LA[sample_index - 12, road_index] * std + mean, LABEL_LA[sample_index, road_index, 0:1] * std + mean],
    axis=-1), marker='P', color='black', linestyle='--', linewidth=1)
plt.plot(range(13, 25, 1), LABEL_LA[sample_index, road_index] * std + mean, marker='P', color='black', linestyle='--',
         linewidth=1, label='Observed')
plt.plot(range(13, 25, 1), DCRNN_LA[sample_index, road_index] * std + mean, marker='P', color='red', linestyle='-',
         linewidth=1, label='DCRNN')
plt.plot(range(13, 25, 1), AGCRN_LA[sample_index, road_index], marker='h', color='blue', linestyle='-', linewidth=1,
         label='AGCRN')
plt.plot(range(13, 25, 1), ASTGCN_LA[sample_index, road_index], marker='o', color='orange', linestyle='-', linewidth=1,
         label='ASTGCN')
plt.plot(range(13, 25, 1), Graph_WaveNet_LA[sample_index, road_index], marker='s', color='#0cdc73', linestyle='-',
         linewidth=1, label='Graph-WaveNet')
plt.plot(range(13, 25, 1), GMAN_LA[sample_index, road_index], marker='p', color='#f504c9', linestyle='-', linewidth=1,
         label='GMAN')
plt.plot(range(13, 25, 1), ST_GRAT_LA[sample_index, road_index], marker='^', color='#d0c101', linestyle='-',
         linewidth=1, label='ST-GRAT')
plt.plot(range(13, 25, 1), MTGNN_LA[sample_index, road_index], marker='d', color='#ff5b00', linestyle='-', linewidth=1,
         label='MTGNN')
plt.plot(range(13, 25, 1), TBLN_LA[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
         label='3S-TBLN')
# plt.legend(loc='upper left', prop=font1)
plt.grid(axis='y')
plt.ylabel('Traffic speed (mile/h)', font2)
# plt.xlabel('Number of samples',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 3)
road_index = 187
plt.plot(range(1, 14, 1), np.concatenate(
    [LABEL_LA[sample_index - 12, road_index] * std + mean, LABEL_LA[sample_index, road_index, 0:1] * std + mean],
    axis=-1), marker='P', color='black', linestyle='--', linewidth=1)
plt.plot(range(13, 25, 1), LABEL_LA[sample_index, road_index] * std + mean, marker='P', color='black', linestyle='--',
         linewidth=1, label='Observed')
plt.plot(range(13, 25, 1), DCRNN_LA[sample_index, road_index] * std + mean, marker='P', color='red', linestyle='-',
         linewidth=1, label='DCRNN')
plt.plot(range(13, 25, 1), AGCRN_LA[sample_index, road_index], marker='h', color='blue', linestyle='-', linewidth=1,
         label='AGCRN')
plt.plot(range(13, 25, 1), ASTGCN_LA[sample_index, road_index], marker='o', color='orange', linestyle='-', linewidth=1,
         label='ASTGCN')
plt.plot(range(13, 25, 1), Graph_WaveNet_LA[sample_index, road_index], marker='s', color='#0cdc73', linestyle='-',
         linewidth=1, label='Graph-WaveNet')
plt.plot(range(13, 25, 1), GMAN_LA[sample_index, road_index], marker='p', color='#f504c9', linestyle='-', linewidth=1,
         label='GMAN')
plt.plot(range(13, 25, 1), ST_GRAT_LA[sample_index, road_index], marker='^', color='#d0c101', linestyle='-',
         linewidth=1, label='ST-GRAT')
plt.plot(range(13, 25, 1), MTGNN_LA[sample_index, road_index], marker='d', color='#ff5b00', linestyle='-', linewidth=1,
         label='MTGNN')
plt.plot(range(13, 25, 1), TBLN_LA[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
         label='3S-TBLN')
# plt.legend(loc='upper left', prop=font1)
plt.grid(axis='y')
plt.ylabel('Traffic speed (mile/h)', font2)
# plt.xlabel('Number of samples',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
# '''





for road_index in range(0):
    print('road_index is : ',road_index)
    plt.figure()
    mean = 54.40592829587626
    std = 19.493739270573094

    plt.plot(
        np.concatenate([list(LABEL_LA[sample_index, road_index] * std + mean) for sample_index in range(12, 1200, 12)],
                       axis=-1), color='black', linestyle='-',
        linewidth=1, label='Observed')
    plt.plot(
        np.concatenate([list(DCRNN_LA[sample_index, road_index] * std + mean) for sample_index in range(12, 1200, 12)],
                       axis=-1), color='red', linestyle='-',
        linewidth=1, label='DCRNN')
    plt.plot(
        np.concatenate([list(AGCRN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)], axis=-1),
        color='blue', linestyle='-', linewidth=1,
        label='AGCRN')
    plt.plot(
        np.concatenate([list(ASTGCN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)], axis=-1),
        color='orange', linestyle='-', linewidth=1,
        label='ASTGCN')
    plt.plot(np.concatenate([list(Graph_WaveNet_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)],
                            axis=-1), color='#0cdc73', linestyle='-',
             linewidth=1, label='Graph-WaveNet')
    plt.plot(np.concatenate([list(GMAN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)], axis=-1),
             color='#f504c9', linestyle='-', linewidth=1,
             label='GMAN')
    plt.plot(
        np.concatenate([list(ST_GRAT_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)], axis=-1),
        color='#d0c101', linestyle='-',
        linewidth=1, label='ST-GRAT')
    plt.plot(
        np.concatenate([list(MTGNN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)], axis=-1),
        color='#ff5b00', linestyle='-', linewidth=1,
        label='MTGNN')
    plt.plot(np.concatenate([list(TBLN_LA[sample_index, road_index]) for sample_index in range(12, 1200, 12)], axis=-1),
             color='#a55af4', linestyle='-', linewidth=1,
             label='3S-TBLN')
    plt.legend(loc='upper left', prop=font1)
    # plt.grid(axis='y')
    plt.ylabel('Traffic speed (km/h)', font2)
    # plt.xlabel('Target time steps', font2)
    # plt.title('Exit tall dataset',font2)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


sample_index = 24
road_index = 1

for sample_index in range(12, 0, 12):
  plt.figure()
  plt.subplot(2, 1, 1)
  road_index = 1
  mean = 54.40592829587626
  std = 19.493739270573094
  plt.plot(range(1,14,1),np.concatenate([LABEL_LA[sample_index-12,road_index]*std + mean,LABEL_LA[sample_index,road_index, 0:1]*std + mean],axis=-1),marker='P',color='black',linestyle='--', linewidth=1,label='Historical Observed')
  plt.plot(range(13,25,1),LABEL_LA[sample_index,road_index]*std + mean,marker='P',color='black',linestyle='-', linewidth=1,label='Observed')
  plt.plot(range(13,25,1),DCRNN_LA[sample_index,road_index]*std + mean,marker='P',color='red',linestyle='-', linewidth=1,label='DCRNN')
  plt.plot(range(13,25,1),AGCRN_LA[sample_index,road_index],marker='h',color='blue',linestyle='-', linewidth=1,label='AGCRN')
  plt.plot(range(13,25,1),ASTGCN_LA[sample_index,road_index],marker='o',color='orange',linestyle='-', linewidth=1,label='ASTGCN')
  plt.plot(range(13,25,1), Graph_WaveNet_LA[sample_index,road_index],marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='Graph-WaveNet')
  plt.plot(range(13,25,1), GMAN_LA[sample_index,road_index],marker='p', color='#f504c9',linestyle='-',linewidth=1,label='GMAN')
  plt.plot(range(13,25,1),ST_GRAT_LA[sample_index,road_index],marker='^',color='#d0c101',linestyle='-', linewidth=1,label='ST-GRAT')
  plt.plot(range(13,25,1),MTGNN_LA[sample_index,road_index],marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='MTGNN')
  plt.plot(range(13, 25, 1), TBLN_LA[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
           label='3S-TBLN')
  plt.legend(loc='upper left',prop=font1)
  plt.grid(axis='y')
  plt.ylabel('Traffic speed (mile/h)',font2)
  plt.xlabel('Target time steps',font2)
  # plt.title('Exit tall dataset',font2)
  plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


  plt.subplot(2, 1, 2)
  road_index = 22
  mean = 80.13087428380071
  std = 30.2782227196378
  plt.plot(range(1,14,1),np.concatenate([LABEL_YC[sample_index-12,road_index]*std + mean,LABEL_YC[sample_index,road_index, 0:1]*std + mean],axis=-1),marker='P',color='black',linestyle='--', linewidth=1,label='Historical Observed')
  plt.plot(range(13,25,1),LABEL_YC[sample_index,road_index]*std + mean,marker='P',color='black',linestyle='-', linewidth=1,label='Observed')
  plt.plot(range(13,25,1),DCRNN_YC[sample_index,road_index]*std + mean,marker='P',color='red',linestyle='-', linewidth=1,label='DCRNN')
  plt.plot(range(13,25,1),AGCRN_YC[sample_index,road_index],marker='h',color='blue',linestyle='-', linewidth=1,label='AGCRN')
  plt.plot(range(13,25,1),ASTGCN_YC[sample_index,road_index],marker='o',color='orange',linestyle='-', linewidth=1,label='ASTGCN')
  plt.plot(range(13,25,1), Graph_WaveNet_YC[sample_index,road_index],marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='Graph-WaveNet')
  plt.plot(range(13,25,1), GMAN_YC[sample_index,road_index],marker='p', color='#f504c9',linestyle='-',linewidth=1,label='GMAN')
  plt.plot(range(13,25,1),ST_GRAT_YC[sample_index,road_index],marker='^',color='#d0c101',linestyle='-', linewidth=1,label='ST-GRAT')
  plt.plot(range(13,25,1),MTGNN_YC[sample_index,road_index],marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='MTGNN')
  plt.plot(range(13, 25, 1), TBLN_YC[sample_index,road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1, label='3S-TBLN')
  plt.legend(loc='upper left',prop=font1)
  plt.grid(axis='y')
  plt.ylabel('Traffic speed (km/h)',font2)
  # plt.xlabel('Number of samples',font2)
  # plt.title('Exit tall dataset',font2)
  plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

  plt.show()






# y=x的拟合可视化图
# '''
begin = 0
total=240

mean = 80.13087428380071
std = 30.2782227196378
LABEL_obs=np.concatenate([list(LABEL_YC[sample_index]*std+mean) for sample_index in range(begin, total, 12)],axis=-1)
DCRNN_pre=np.concatenate([list(DCRNN_YC[sample_index]*std + mean) for sample_index in range(begin, total, 12)],axis=-1)
AGCRN_pre=np.concatenate([list(AGCRN_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
Graph_WaveNet_pre=np.concatenate([list(Graph_WaveNet_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
GMAN_pre=np.concatenate([list(GMAN_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
ASTGCN_pre=np.concatenate([list(ASTGCN_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
TBLN_pre=np.concatenate([list(TBLN_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
print(LABEL_obs.shape, TBLN_pre.shape)
# plt.figure()
plt.subplot(2,3,1)
plt.scatter(LABEL_obs,DCRNN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'DCRNN',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
plt.ylabel("Predicted traffic speed", font2)
plt.legend(loc='lower right',prop=font2)

plt.subplot(2,3,2)
plt.scatter(LABEL_obs,AGCRN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'AGCRN',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
plt.legend(loc='lower right',prop=font2)

plt.subplot(2,3,3)
plt.scatter(LABEL_obs,Graph_WaveNet_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'Graph-WaveNet',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (ug/m3)", font2)
# plt.ylabel("Predicted traffic spedd", font2)
plt.legend(loc='lower right',prop=font2)

plt.subplot(2,3,4)
plt.scatter(LABEL_obs,GMAN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'GMAN',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic speed", font2)
plt.ylabel("Predicted traffic speed", font2)
plt.legend(loc='lower right',prop=font2)

plt.subplot(2,3,5)
plt.scatter(LABEL_obs,ASTGCN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'ASTGCN',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Exit tall dataset", font2)
plt.xlabel("Observed traffic speed", font2)
plt.legend(loc='lower right',prop=font2)

plt.subplot(2,3,6)
plt.scatter(LABEL_obs,TBLN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'3S-TBLN',linewidths=1)
c=[i for i in range(150)]
d=[i for i in range(150)]
plt.plot(c,d,'black',linewidth=2)
# plt.title("Gantry dataset", font2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic speed", font2)
plt.legend(loc='lower right',prop=font2)
plt.show()
# '''


import matplotlib.gridspec as gridspec
sns.set_theme(style='ticks', font_scale=1.5,font='Times New Roman')

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(DCRNN_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df)
g.set_axis_labels(xlabel='Observed traffic speed (km/h)', ylabel='Predicted traffic speed (km/h)')
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(AGCRN_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df)
# g = sns.jointplot(data=MT_STAN, x="segment_label_"+str(i), y="segment_pre_"+str(i), hue="vehicle type", kind="kde")
g.set_axis_labels(xlabel='Observed traffic speed (km/h)', ylabel='Predicted traffic speed (km/h)')
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(Graph_WaveNet_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df)
# g = sns.jointplot(data=MT_STAN, x="segment_label_"+str(i), y="segment_pre_"+str(i), hue="vehicle type", kind="kde")
g.set_axis_labels(xlabel='Observed traffic speed (km/h)', ylabel='Predicted traffic speed (km/h)')
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(GMAN_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df)
# g = sns.jointplot(data=MT_STAN, x="segment_label_"+str(i), y="segment_pre_"+str(i), hue="vehicle type", kind="kde")
g.set_axis_labels(xlabel='Observed traffic speed (km/h)', ylabel='Predicted traffic speed (km/h)')
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(ASTGCN_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df)
# g = sns.jointplot(data=MT_STAN, x="segment_label_"+str(i), y="segment_pre_"+str(i), hue="vehicle type", kind="kde")
g.set_axis_labels(xlabel='Observed traffic speed (km/h)', ylabel='Predicted traffic speed (km/h)')
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(TBLN_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df)
# g = sns.jointplot(data=MT_STAN, x="segment_label_"+str(i), y="segment_pre_"+str(i), hue="vehicle type", kind="kde")
g.set_axis_labels(xlabel='Observed traffic speed (km/h)', ylabel='Predicted traffic speed (km/h)')
plt.show()