# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from models.inits import *
from matplotlib.ticker import MaxNLocator
import pandas as pd

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14.,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

baseline = [6.236,6.074,6.499,6.368,6.112,5.809,5.868,5.604,5.462,5.375,5.752,5.357,5.403,5.943,5.613,5.225]
print([round(char, 3) for char in (np.array(baseline)-5.225)/np.array(baseline)*100])


baseline = [6.131,5.893,6.067,5.949,5.824,5.644,5.726,5.442,5.346,5.298,5.571,5.264,5.350,5.623,5.514,5.154]
print([round(char, 3) for char in (np.array(baseline)-5.154)/np.array(baseline)*100])



LSTM_BILSTM_MAE_1=[5.514, 5.634, 5.75, 5.869, 5.968, 6.064, 6.152, 6.234, 6.304, 6.374, 6.442, 6.499]
MDL_MAE_1=[5.343, 5.48, 5.558, 5.592, 5.62, 5.656, 5.692, 5.72, 5.741, 5.759, 5.764, 5.809]
T_GCN_MAE_1 = [5.651, 5.647, 5.648, 5.669, 5.687, 5.7, 5.723, 5.744, 5.763, 5.791, 5.821, 5.868]
STGCN_MAE_1 = [5.148, 5.244, 5.306, 5.364, 5.408, 5.455, 5.49, 5.53, 5.557, 5.6, 5.6, 5.604]
DCRNN_MAE_1=[5.196, 5.243, 5.277, 5.307, 5.328, 5.346, 5.363, 5.379, 5.397, 5.416, 5.438, 5.462]
Graph_WaveNet_MAE_1=[5.109, 5.174, 5.21, 5.231, 5.254, 5.267, 5.287, 5.305, 5.308, 5.325, 5.339, 5.357]
ASTGCN_MAE_1=[5.302, 5.388, 5.453, 5.516, 5.561, 5.583, 5.615, 5.633, 5.659, 5.675, 5.714, 5.752]
AGCRN_MAE_1=[5.179, 5.179, 5.247, 5.259, 5.307, 5.308, 5.361, 5.322, 5.319, 5.331, 5.383, 5.375]
GMAN_MAE_1=[5.351, 5.317, 5.318, 5.322, 5.326, 5.333, 5.345, 5.354, 5.364, 5.379, 5.391, 5.403]
MTGNN_MAE_1=[5.406, 5.42, 5.441, 5.461, 5.489, 5.512, 5.532, 5.558, 5.562, 5.581, 5.593, 5.613]
ST_GRAT_MAE_1=[5.176, 5.389, 5.556, 5.592, 5.54, 5.544, 5.623, 5.7, 5.75, 5.799, 5.868, 5.943]
TBLN_MAE_1=[5.007, 5.046, 5.062, 5.074, 5.089, 5.1, 5.111, 5.121, 5.135, 5.148, 5.166, 5.18]
STGIN_MAE_1=[5.102, 5.113, 5.119, 5.129, 5.133, 5.142, 5.15, 5.162, 5.173, 5.188, 5.207, 5.225]

LSTM_BILSTM_RMSE_1=[5.514, 5.634, 5.75, 5.869, 5.968, 6.064, 6.152, 6.234, 6.304, 6.374, 6.442, 6.499]
MDL_RMSE_1=[8.911, 9.086, 9.181, 9.214, 9.242, 9.282, 9.323, 9.356, 9.378, 9.399, 9.376, 9.433]
T_GCN_RMSE_1 = [9.25, 9.245, 9.235, 9.265, 9.288, 9.308, 9.334, 9.355, 9.379, 9.41, 9.445, 9.496]
STGCN_RMSE_1 = [8.733, 8.826, 8.887, 8.953, 8.986, 9.022, 9.057, 9.102, 9.096, 9.141, 9.146, 9.166]
DCRNN_RMSE_1=[8.736, 8.814, 8.86, 8.899, 8.923, 8.946, 8.963, 8.98, 9.002, 9.024, 9.045, 9.065]
Graph_WaveNet_RMSE_1=[8.701, 8.765, 8.804, 8.825, 8.86, 8.866, 8.884, 8.907, 8.899, 8.928, 8.944, 8.954]
ASTGCN_RMSE_1=[8.877, 8.965, 9.014, 9.075, 9.133, 9.172, 9.192, 9.214, 9.223, 9.227, 9.256, 9.29]
AGCRN_RMSE_1=[8.861, 8.788, 8.923, 8.922, 8.999, 8.964, 9.051, 8.948, 8.961, 8.986, 9.124, 9.027]
GMAN_RMSE_1=[8.942, 8.922, 8.93, 8.938, 8.943, 8.951, 8.963, 8.972, 8.983, 8.999, 9.014, 9.032]
MTGNN_RMSE_1=[9.019, 9.041, 9.071, 9.101, 9.124, 9.158, 9.185, 9.215, 9.226, 9.239, 9.253, 9.268]
ST_GRAT_RMSE_1=[8.791, 9.04, 9.216, 9.246, 9.167, 9.107, 9.119, 9.166, 9.222, 9.29, 9.374, 9.464]
TBLN_RMSE_1=[8.673, 8.795, 8.813, 8.843, 8.867, 8.877, 8.886, 8.886, 8.893, 8.906, 8.928, 8.934]
STGIN_RMSE_1=[8.878, 8.896, 8.899, 8.916, 8.924, 8.935, 8.942, 8.954, 8.961, 8.98, 8.994, 9.001]

LSTM_BILSTM_MAPE_1=[19.593, 19.951, 20.106, 20.26, 20.367, 20.485, 20.596, 20.692, 20.782, 20.861, 20.946, 21.021]
MDL_MAPE_1=[13.784, 12.762, 12.798, 12.829, 12.712, 12.677, 12.718, 12.77, 12.855, 13.014, 13.0, 13.2]
T_GCN_MAPE_1 = [12.733, 12.645, 12.698, 12.738, 12.909, 12.839, 12.882, 13.174, 13.037, 13.201, 13.243, 13.664]
STGCN_MAPE_1 = [11.912, 12.329, 12.401, 12.577, 12.501, 12.716, 12.692, 12.791, 13.019, 13.151, 12.964, 12.907]
DCRNN_MAPE_1=[11.827, 11.828, 11.917, 11.938, 11.959, 11.977, 12.005, 12.042, 12.08, 12.126, 12.175, 12.224]
Graph_WaveNet_MAPE_1=[11.813, 12.358, 12.448, 11.919, 12.023, 12.232, 12.248, 12.037, 12.193, 12.249, 12.372, 12.102]
ASTGCN_MAPE_1=[13.007, 12.98, 13.222, 13.073, 13.08, 13.185, 13.338, 13.281, 13.353, 13.329, 13.716, 13.671]
AGCRN_MAPE_1=[11.796, 11.671, 12.363, 11.829, 12.822, 12.493, 13.272, 12.961, 11.903, 12.053, 12.467, 13.116]
GMAN_MAPE_1=[13.078, 12.833, 12.762, 12.717, 12.708, 12.679, 12.655, 12.647, 12.657, 12.667, 12.736, 12.757]
MTGNN_MAPE_1=[13.001, 13.107, 13.467, 13.447, 13.865, 13.644, 13.861, 13.944, 13.421, 13.884, 14.048, 14.324]
ST_GRAT_MAPE_1=[12.311, 12.961, 13.296, 13.328, 13.174, 13.099, 13.136, 13.153, 13.085, 12.994, 13.014, 13.182]
TBLN_MAPE_1=[11.174, 11.396, 11.264, 11.061, 11.098, 11.131, 11.173, 11.195, 11.249, 11.294, 11.386, 11.412]
STGIN_MAPE_1=[11.651, 11.611, 11.618, 11.623, 11.618, 11.622, 11.601, 11.63, 11.62, 11.629, 11.668, 11.666]

'''
data = pd.read_csv('metrics.csv').values
print(data.shape)

mae,rmse,mape =[],[],[]
for i in range(12):
    list_ = data[i,0].split()
    print(list_)
    mae.append(float(list_[-3]))
    rmse.append(float(list_[-2]))
    mape.append(float(list_[-1][:-1]))

print(mae)
print(rmse)
print(mape)
'''


LSTM_BILSTM_MAE_2=[2.462, 2.82, 3.093, 3.337, 3.563, 3.782, 3.989, 4.189, 4.38, 4.566, 4.747, 4.927]
MDL_MAE_2=[2.466, 2.786, 3.006, 3.178, 3.328, 3.464, 3.587, 3.695, 3.796, 3.894, 4.012, 4.119]
DCRNN_MAE_2=[2.248, 2.539, 2.737, 2.898, 3.033, 3.154, 3.264, 3.363, 3.454, 3.54, 3.622, 3.706]
Graph_WaveNet_MAE_2=[2.232, 2.509, 2.699, 2.854, 2.98, 3.105, 3.212, 3.305, 3.389, 3.457, 3.518, 3.576]
ASTGCN_MAE_2=[2.423, 2.753, 2.994, 3.203, 3.379, 3.517, 3.652, 3.771, 3.882, 3.988, 4.086, 4.182]
AGCRN_MAE_2=[2.405, 2.664, 2.857, 3.009, 3.125, 3.218, 3.302, 3.374, 3.436, 3.493, 3.557, 3.624]
GMAN_MAE_2=[2.46, 2.676, 2.83, 2.955, 3.058, 3.146, 3.222, 3.286, 3.342, 3.393, 3.439, 3.483]
MTGNN_MAE_2=[2.445, 2.695, 2.87, 3.014, 3.139, 3.247, 3.339, 3.424, 3.504, 3.58, 3.656, 3.736]
ST_GRAT_MAE_2=[2.117, 2.49, 2.733, 2.925, 3.095, 3.248, 3.391, 3.523, 3.65, 3.773, 3.897, 4.027]
TBLN_MAE_2=[2.306, 2.558, 2.727, 2.861, 2.979, 3.076, 3.159, 3.232, 3.295, 3.347, 3.392, 3.434]

LSTM_BILSTM_RMSE_2=[4.381, 5.386, 6.106, 6.698, 7.214, 7.682, 8.111, 8.504, 8.865, 9.209, 9.531, 9.842]
MDL_RMSE_2=[4.295, 5.202, 5.819, 6.272, 6.642, 6.962, 7.241, 7.486, 7.704, 7.905, 8.097, 8.231]
DCRNN_RMSE_2=[3.921, 4.773, 5.353, 5.811, 6.182, 6.503, 6.783, 7.029, 7.247, 7.449, 7.637, 7.822]
Graph_WaveNet_RMSE_2=[3.826, 4.596, 5.11, 5.535, 5.878, 6.202, 6.481, 6.724, 6.934, 7.101, 7.245, 7.374]
ASTGCN_RMSE_2=[4.208, 5.09, 5.741, 6.29, 6.732, 7.027, 7.299, 7.536, 7.753, 7.957, 8.141, 8.311]
AGCRN_RMSE_2=[4.233, 5.006, 5.509, 5.921, 6.283, 6.544, 6.758, 6.925, 7.072, 7.226, 7.372, 7.523]
GMAN_RMSE_2=[4.439, 5.13, 5.599, 5.97, 6.265, 6.515, 6.728, 6.903, 7.05, 7.179, 7.293, 7.396]
MTGNN_RMSE_2=[4.214, 4.951, 5.459, 5.859, 6.19, 6.472, 6.719, 6.925, 7.118, 7.295, 7.462, 7.631]
ST_GRAT_RMSE_2=[3.721, 4.698, 5.363, 5.888, 6.32, 6.694, 7.033, 7.343, 7.648, 7.937, 8.214, 8.489]
TBLN_RMSE_2=[4.021, 4.785, 5.278, 5.678, 6.016, 6.288, 6.519, 6.716, 6.877, 7.007, 7.118, 7.22]

LSTM_BILSTM_MAPE_2=[6.255, 7.458, 8.373, 9.196, 9.95, 10.678, 11.37, 12.031, 12.663, 13.275, 13.873, 14.479]
MDL_MAPE_2=[6.067, 7.152, 7.996, 8.653, 9.247, 9.777, 10.259, 10.695, 11.1, 11.487, 11.913, 12.392]
DCRNN_MAPE_2=[5.44, 6.408, 7.112, 7.716, 8.223, 8.658, 9.042, 9.388, 9.708, 10.003, 10.279, 10.553]
Graph_WaveNet_MAPE_2=[5.334, 6.27, 6.988, 7.557, 8.058, 8.482, 8.88, 9.266, 9.573, 9.868, 10.111, 10.328]
ASTGCN_MAPE_2=[6.041, 7.188, 8.084, 8.805, 9.447, 9.893, 10.284, 10.813, 11.129, 11.371, 11.635, 11.92]
AGCRN_MAPE_2=[6.027, 6.914, 7.617, 8.22, 8.663, 9.016, 9.257, 9.516, 9.798, 10.008, 10.228, 10.485]
GMAN_MAPE_2=[6.124, 6.886, 7.481, 7.986, 8.407, 8.766, 9.078, 9.342, 9.566, 9.768, 9.955, 10.137]
MTGNN_MAPE_2=[5.991, 6.804, 7.403, 7.941, 8.384, 8.786, 9.166, 9.508, 9.836, 10.131, 10.399, 10.68]
ST_GRAT_MAPE_2=[5.042, 6.261, 7.148, 7.861, 8.465, 8.968, 9.408, 9.784, 10.139, 10.472, 10.799, 11.135]
TBLN_MAPE_2=[5.704, 6.514, 7.149, 7.649, 8.105, 8.478, 8.794, 9.093, 9.338, 9.541, 9.731, 9.913]


'''
LSTM = pd.read_csv('results/LSTM.csv',encoding='utf-8').values[108:]
Bi_LSTM = pd.read_csv('results/BILSTM.csv',encoding='utf-8').values[108:]
FI_RNNs = pd.read_csv('results/FI-RNN.csv',encoding='utf-8').values[108:]
GMAN = pd.read_csv('results/GMAN.csv',encoding='utf-8').values[108:]
STGIN = pd.read_csv('results/STGIN.csv',encoding='utf-8').values
PSPNN = pd.read_csv('results/PSPNN.csv',encoding='utf-8').values[108:]
MDL = pd.read_csv('results/MDL.csv',encoding='utf-8').values[108:]
AST_GAT=pd.read_csv('results/AST-GAT.csv',encoding='utf-8').values[108:]
T_GCN=pd.read_csv('results/T-GCN.csv',encoding='utf-8').values[108:]
RST = pd.read_csv('results/RST.csv',encoding='utf-8').values[108:]

LSTM_pre = []
LSTM_obs = []
Bi_LSTM_pre = []
Bi_LSTM_obs = []
FI_RNNs_pre = []
FI_RNNs_obs = []
GMAN_pre = []
GMAN_obs = []
STGIN_pre = []
STGIN_obs = []
PSPNN_pre = []
PSPNN_obs = []
MDL_pre = []
MDL_obs = []
AST_GAT_pre = []
AST_GAT_obs = []
T_GCN_pre = []
T_GCN_obs = []
RST_pre = []
RST_obs = []

K = 10
site_num=108
for i in range(site_num,site_num*K,site_num):
    LSTM_obs.append(LSTM[i:i+site_num,19:25])
    LSTM_pre.append(LSTM[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    Bi_LSTM_obs.append(Bi_LSTM[i:i+site_num,19:25])
    Bi_LSTM_pre.append(Bi_LSTM[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    FI_RNNs_obs.append(FI_RNNs[i:i+site_num,19:25])
    FI_RNNs_pre.append(FI_RNNs[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    GMAN_obs.append(GMAN[i:i+site_num,19:25])
    GMAN_pre.append(GMAN[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    STGIN_obs.append(STGIN[i:i+site_num,19:25])
    STGIN_pre.append(STGIN[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    PSPNN_obs.append(PSPNN[i:i+site_num,19:25])
    PSPNN_pre.append(PSPNN[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    MDL_obs.append(MDL[i:i+site_num,19:25])
    MDL_pre.append(MDL[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    AST_GAT_obs.append(AST_GAT[i:i+site_num,19:25])
    AST_GAT_pre.append(AST_GAT[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    T_GCN_obs.append(T_GCN[i:i+site_num,19:25])
    T_GCN_pre.append(T_GCN[i:i + site_num, 25:])

for i in range(site_num,site_num*K,site_num):
    RST_obs.append(RST[i:i+site_num,19:25])
    RST_pre.append(RST[i:i + site_num, 25:])
'''


'''
plt.subplot(3, 1, 1)
i,j=8,4
print(STGIN_obs[i][j])
plt.xticks(range(1, 7), ['2021.8.14 6:45', '7:00', '7:15', '7:30', '7:45', '8:00'])
plt.plot(range(1, 7), STGIN_obs[i][j], marker='d', color='blue', label=u'Observed', linewidth=1)
plt.plot(range(1, 7), STGIN_pre[i][j], marker='X', color='#a55af4', label=u'STGIN', linewidth=1)
plt.plot(range(1, 7), MDL_pre[i][j], marker='p', color='#f504c9', label=u'MDL', linewidth=1)
plt.plot(range(1, 7), GMAN_pre[i][j], marker='^', color='#d0c101', label=u'GMAN', linewidth=1)
plt.plot(range(1, 7), AST_GAT_pre[i][j], marker='*', color='#82cafc', label=u'AST-GAT', linewidth=1)
plt.plot(range(1, 7), PSPNN_pre[i][j], marker='s', color='#0cdc73', label=u'PSPNN', linewidth=1)
plt.plot(range(1, 7), T_GCN_pre[i][j], marker='d', color='#ff5b00', label=u'T-GCN', linewidth=1)
plt.legend(loc='upper left', prop=font2)
# plt.xlabel("Target time steps", font2)
plt.ylabel("Taffic speed", font1)
plt.title("Road segment 1", font1)

i,j=8,10
print(STGIN_obs[i][j])
plt.subplot(3, 1, 2)
# i,j=1,0
plt.xticks(range(1, 7), ['2021.8.14 6:45', '7:00', '7:15', '7:30', '7:45', '8:00'])
plt.plot(range(1, 7), STGIN_obs[i][j], marker='d', color='blue', label=u'Observed', linewidth=1)
plt.plot(range(1, 7), STGIN_pre[i][j], marker='X', color='#a55af4', label=u'STGIN', linewidth=1)
plt.plot(range(1, 7), MDL_pre[i][j], marker='p', color='#f504c9', label=u'MDL', linewidth=1)
plt.plot(range(1, 7), GMAN_pre[i][j], marker='^', color='#d0c101', label=u'GMAN', linewidth=1)
plt.plot(range(1, 7), PSPNN_pre[i][j], marker='s', color='#0cdc73', label=u'PSPNN', linewidth=1)
plt.plot(range(1, 7), T_GCN_pre[i][j], marker='d', color='#ff5b00', label=u'T-GCN', linewidth=1)
plt.plot(range(1, 7), AST_GAT_pre[i][j], marker='*', color='#82cafc', label=u'AST-GAT', linewidth=1)
# plt.legend(loc='upper left', prop=font2)
# plt.xlabel("Target time steps", font2)
plt.ylabel("Taffic speed", font1)
plt.title("Road segment 2", font1)

i,j=8,97
print(STGIN_obs[i][j])
plt.subplot(3, 1, 3)
# i,j=1,0
plt.xticks(range(1, 7), ['2021.8.14 6:45', '7:00', '7:15', '7:30', '7:45', '8:00'])
plt.plot(range(1, 7), STGIN_obs[i][j], marker='d', color='blue', label=u'Observed', linewidth=1)
plt.plot(range(1, 7), STGIN_pre[i][j], marker='X', color='#a55af4', label=u'STGIN', linewidth=1)
plt.plot(range(1, 7), MDL_pre[i][j], marker='p', color='#f504c9', label=u'MDL', linewidth=1)
plt.plot(range(1, 7), GMAN_pre[i][j], marker='^', color='#d0c101', label=u'GMAN', linewidth=1)
plt.plot(range(1, 7), AST_GAT_pre[i][j], marker='*', color='#82cafc', label=u'AST-GAT', linewidth=1)
plt.plot(range(1, 7), PSPNN_pre[i][j], marker='s', color='#0cdc73', label=u'PSPNN', linewidth=1)
plt.plot(range(1, 7), T_GCN_pre[i][j], marker='d', color='#ff5b00', label=u'T-GCN', linewidth=1)
# plt.legend(loc='upper left', prop=font2)
# plt.xlabel("Target time steps", font2)
plt.ylabel("Taffic speed", font1)
plt.title("Road segment 3", font1)

plt.show()
'''


'''
for i in range(8, len(STGIN_pre)):
    for j in range(108):
        print(i, j)
        # plt.figure()
        plt.subplot(1,1,1)
        # i,j=1,0
        plt.xticks(range(1,7), ['2021.8.26 7:50','7:55','8:00','8:05','8:10','8:15'])
        plt.plot(range(1,7),STGIN_obs[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
        plt.plot(range(1,7),STGIN_pre[i][j],marker='o', color= 'red', label=u'STGIN', linewidth=1)
        plt.plot(range(1,7),GMAN_pre[i][j],marker='s', color= '#d0c101', label=u'GMAN', linewidth=1)
        plt.plot(range(1,7),AST_GAT_pre[i][j],marker='s', color= '#82cafc', label=u'AST_GAT', linewidth=1)
        plt.plot(range(1, 7), PSPNN_pre[i][j], marker='s', color='#0cdc73', label=u'PSPNN', linewidth=1)
        plt.plot(range(1,7),T_GCN_pre[i][j],marker='s', color= 'blue', label=u'T_GCN', linewidth=1)
        plt.plot(range(1, 7), RST_pre[i][j], marker='s', color='orange', label=u'RST', linewidth=1)
        plt.legend(loc='upper left',prop=font2)
        # plt.xlabel("Target time steps", font2)
        plt.ylabel("Taffic speed", font2)
        # plt.title("Entrance toll dataset (sample 1)", font2)


        # plt.subplot(2,1,2)
        # i,j=10,16
        # plt.xticks(range(1,7), ['2021.8.26 7:50','7:55','8:00','8:05','8:10','8:15'])
        # plt.plot(range(1,7),STGIN_obs[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
        # plt.plot(range(1,7),STGIN_pre[i][j],marker='o', color= 'orange', label=u'STGIN', linewidth=1)
        # plt.plot(range(1,7),GMAN_pre[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
        # plt.plot(range(1,7),AST_GAT_pre[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
        # plt.plot(range(1,7),T_GCN_pre[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
        # plt.legend(loc='upper left',prop=font2)
        # plt.xlabel("Target time steps", font2)
        # plt.ylabel("Taffic speed", font2)
        # # plt.title("Gantry dataset (sample 2)", font2)
        plt.show()
'''
#
# plt.subplot(6,1,2)
# i,j=10,3
# plt.xticks(range(1,7), ['7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_2[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_2[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_2[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# # plt.xlabel("Target time steps", font2)
# # plt.ylabel("Taffic flow", font2)
# plt.title("Exit toll dataset (sample 1)", font2)
#
# plt.subplot(6,1,3)
# i,j=10,39
# plt.xticks(range(1,7), ['7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_3[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_3[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_3[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# # plt.xlabel("Target time steps", font2)
# # plt.ylabel("Taffic flow", font2)
# plt.title("Gantry dataset (sample 1)", font2)

# plt.subplot(6,1,4)
# i,j=10,4
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xticks(range(1,7), ['2021.8.26 7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),STGIN[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_1[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_1[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# plt.xlabel("Target time steps", font2)
# plt.ylabel("Taffic flow", font2)
# plt.title("Entrance toll dataset (sample 2)", font2)

# plt.subplot(6,1,5)
# i,j=10,6
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xticks(range(1,7), ['7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_2[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_2[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_2[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# plt.xlabel("Target time steps", font2)
# # plt.ylabel("Taffic flow", font2)
# plt.title("Exit toll dataset (sample 2)", font2)



# y=x的拟合可视化图
'''
# plt.figure()
plt.subplot(3,3,1)
plt.scatter(LSTM_obs,LSTM_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'LSTM',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (ug/m3)", font2)
plt.ylabel("Predicted traffic spedd", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,2)
plt.scatter(Bi_LSTM_obs,Bi_LSTM_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'Bi-LSTM',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (ug/m3)", font2)
# plt.ylabel("Predicted traffic spedd", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,3)
plt.scatter(FI_RNNs_obs,FI_RNNs_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'FI-RNNs',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (ug/m3)", font2)
# plt.ylabel("Predicted traffic spedd", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,4)
plt.scatter(PSPNN_obs,PSPNN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'PSPNN',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (ug/m3)", font2)
plt.ylabel("Predicted traffic spedd", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,5)
plt.scatter(MDL_obs,MDL_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'MDL',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Exit tall dataset", font2)
# plt.xlabel("Observed PM2.5 (μg/m3)", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,6)
plt.scatter(T_GCN_obs,T_GCN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'T-GCN',linewidths=1)
c=[i for i in range(150)]
d=[i for i in range(150)]
plt.plot(c,d,'black',linewidth=2)
# plt.title("Gantry dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (μg/m3)", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,7)
plt.scatter(AST_GAT_obs,AST_GAT_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'AST-GAT',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic speed", font2)
plt.ylabel("Predicted traffic speed", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,8)
plt.scatter(GMAN_obs,GMAN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'GMAN',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic speed", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(3,3,9)
plt.scatter(STGIN_obs,STGIN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'STGIN',linewidths=1)
plt.plot(c,d,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic speed", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)
plt.show()
'''


# 可视化每个模型在MAPE上的一个表现，柱状图
'''
x = np.arange(1, 8, 1)
total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2
plt.subplot(1,2,1)
rmse_1=[5.6085,5.5612,5.7057,5.8235,5.6626,6.0323,5.5336]
rmse_2=[5.3883,5.4198,5.5582,5.6798,5.5089,5.7972,5.3924]
rmse_3=[7.4951,7.4619,7.5817,7.5810,7.4643,7.8891,7.4103]
mape_1=[0.3756,0.3498,0.4027,0.3600,0.3527,0.3900,0.3516]
mape_2=[0.3545,0.3255,0.3665,0.3464,0.3340,0.3653,0.3282]
mape_3=[0.2836,0.2789,0.2810,0.2735,0.2685,0.2871,0.2765]
plt.ylim(4,8)
plt.xticks(range(1,9),['GMAN','MT-STNet','STNet','STNet-1','STNet-2','STNet-3','STNet-4'])
plt.bar(x, rmse_1, width=width,label='Entrance toll dataset',color = 'red')
plt.bar(x + width, rmse_2, width=width,label='Exit toll dataset',color = 'black')
plt.bar(x + 2 * width, rmse_3, width=width,label='Gantry dataset',color='salmon')
plt.ylabel('RMSE',font2)
# plt.title('Target time steps $Q$ = 6 ([0-30 min])',font2)
plt.legend()

plt.subplot(1,2,2)
plt.ylim(0.2, 0.45)
plt.xticks(range(1,9),['GMAN','MT-STNet','STNet','STNet-1','STNet-2','STNet-3','STNet-4'])
plt.bar(x, mape_1, width=width,label='Entrance toll dataset',color = 'red')
plt.bar(x + width, mape_2, width=width,label='Exit toll dataset',color = 'black')
plt.bar(x + 2 * width, mape_3, width=width,label='Gantry dataset',color='salmon')
plt.ylabel('MAPE',font2)
# plt.title('Target time steps $Q$ = 6 ([0-30 min])',font2)
plt.legend()
plt.show()
'''

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# No decimal placesplt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

# 可视化每个模型在MAE，RMSE和MAPE上的一个表现
'''
ax=plt.subplot(2,3,1)
plt.plot(range(1,13,1),DCRNN_MAE_1,marker='P',color='red',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,13,1),AGCRN_MAE_1,marker='h',color='blue',linestyle='-', linewidth=1,label='AGCRN')
plt.plot(range(1,13,1),ASTGCN_MAE_1,marker='o',color='orange',linestyle='-', linewidth=1,label='ASTGCN')
plt.plot(range(1,13,1), Graph_WaveNet_MAE_1,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='Graph-WaveNet')
plt.plot(range(1,13,1), GMAN_MAE_1,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='GMAN')
plt.plot(range(1,13,1),ST_GRAT_MAE_1,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,13,1),MTGNN_MAE_1 ,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='MTGNN')
# plt.plot(range(1,7,1), AST_GAT_mae,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,13,1), TBLN_MAE_1,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='3S-TBLN')
ymajorFormatter = FormatStrFormatter('%1.1f')
ax.yaxis.set_major_formatter(ymajorFormatter)
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
# plt.ylabel('MAE',font2)
# plt.xlabel('Target time steps',font2)
plt.title('MAE',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(2,3,2)
# plt.xticks(range(1,8), range(0,31,5))
plt.plot(range(1,13,1),DCRNN_RMSE_1,marker='P',color='red',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,13,1),AGCRN_RMSE_1,marker='h',color='blue',linestyle='-', linewidth=1,label='AGCRN')
plt.plot(range(1,13,1),ASTGCN_RMSE_1,marker='o',color='orange',linestyle='-', linewidth=1,label='ASTGCN')
plt.plot(range(1,13,1), Graph_WaveNet_RMSE_1,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='Graph-WaveNet')
plt.plot(range(1,13,1), GMAN_RMSE_1,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='GMAN')
plt.plot(range(1,13,1),ST_GRAT_RMSE_1,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,13,1),MTGNN_RMSE_1,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='MTGNN')
# plt.plot(range(1,7,1), AST_GAT_rmse,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,13,1), TBLN_RMSE_1,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='3S-TBLN')
# plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
# plt.ylabel('RMSE',font2)
# plt.xlabel('Target time steps',font2)
plt.title('RMSE',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(2,3,3)
plt.plot(range(1,13,1),DCRNN_MAPE_1,marker='P',color='red',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,13,1),AGCRN_MAPE_1,marker='h',color='blue',linestyle='-', linewidth=1,label='AGCRN')
plt.plot(range(1,13,1),ASTGCN_MAPE_1,marker='o',color='orange',linestyle='-', linewidth=1,label='ASTGCN')
plt.plot(range(1,13,1), Graph_WaveNet_MAPE_1,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='Graph-WaveNet')
plt.plot(range(1,13,1), GMAN_MAPE_1,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='GMAN')
plt.plot(range(1,13,1),ST_GRAT_MAPE_1,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,13,1),MTGNN_MAPE_1,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='MTGNN')
# plt.plot(range(1,7,1), AST_GAT_mape,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,13,1), TBLN_MAPE_1,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='3S-TBLN')
# plt.ylabel('MAPE',font2)
# plt.xlabel('Target time steps',font2)
plt.title('MAPE (%)',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
# plt.show()

plt.subplot(2,3,4)
plt.plot(range(1,13,1),DCRNN_MAE_2,marker='P',color='red',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,13,1),AGCRN_MAE_2,marker='h',color='blue',linestyle='-', linewidth=1,label='AGCRN')
plt.plot(range(1,13,1),ASTGCN_MAE_2,marker='o',color='orange',linestyle='-', linewidth=1,label='ASTGCN')
plt.plot(range(1,13,1), Graph_WaveNet_MAE_2,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='Graph-WaveNet')
plt.plot(range(1,13,1), GMAN_MAE_2,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='GMAN')
plt.plot(range(1,13,1),ST_GRAT_MAE_2,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,13,1),MTGNN_MAE_2,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='MTGNN')
# plt.plot(range(1,7,1), AST_GAT_mae,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,13,1), TBLN_MAE_2,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='3S-TBLN')
# plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
# plt.ylabel('MAE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('MAE',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

ax=plt.subplot(2,3,5)
# plt.xticks(range(1,8), range(0,31,5))
plt.plot(range(1,13,1),DCRNN_RMSE_2,marker='P',color='red',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,13,1),AGCRN_RMSE_2,marker='h',color='blue',linestyle='-', linewidth=1,label='AGCRN')
plt.plot(range(1,13,1),ASTGCN_RMSE_2,marker='o',color='orange',linestyle='-', linewidth=1,label='ASTGCN')
plt.plot(range(1,13,1), Graph_WaveNet_RMSE_2,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='Graph-WaveNet')
plt.plot(range(1,13,1), GMAN_RMSE_2,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='GMAN')
plt.plot(range(1,13,1),ST_GRAT_RMSE_2,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,13,1),MTGNN_RMSE_2,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='MTGNN')
# plt.plot(range(1,7,1), AST_GAT_rmse,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,13,1), TBLN_RMSE_2,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='3S-TBLN')
ymajorFormatter = FormatStrFormatter('%1.1f')
ax.yaxis.set_major_formatter(ymajorFormatter)
# plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
# plt.ylabel('RMSE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

ax=plt.subplot(2,3,6)
plt.plot(range(1,13,1),DCRNN_MAPE_2,marker='P',color='red',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,13,1),AGCRN_MAPE_2,marker='h',color='blue',linestyle='-', linewidth=1,label='AGCRN')
plt.plot(range(1,13,1),ASTGCN_MAPE_2,marker='o',color='orange',linestyle='-', linewidth=1,label='ASTGCN')
plt.plot(range(1,13,1), Graph_WaveNet_MAPE_2,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='Graph-WaveNet')
plt.plot(range(1,13,1), GMAN_MAPE_2,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='GMAN')
plt.plot(range(1,13,1),ST_GRAT_MAPE_2,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,13,1),MTGNN_MAPE_2,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='MTGNN')
# plt.plot(range(1,7,1), AST_GAT_mape,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,13,1), TBLN_MAPE_2,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='3S-TBLN')
ymajorFormatter = FormatStrFormatter('%1.1f')
ax.yaxis.set_major_formatter(ymajorFormatter)
# plt.ylabel('MAPE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('Gantry dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.show()
'''

yinchuan = pd.read_csv('/Users/guojianzou/Traffic-speed-prediction/3S-TBLN/data/yinchuan/train_15.csv')

metr = pd.read_csv('/Users/guojianzou/Traffic-speed-prediction/3S-TBLN/data/metr-la/train_5.csv')

'''
# 显示周期性
weeks = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.figure()
plt.subplot(1,2,1)
xlabel = [i for i in range(0, 672,96)]
plt.plot(yinchuan[yinchuan['node']==1].values[:672,-1],color='#ff5b00',linestyle='-',linewidth=1,label='1st week')
plt.plot(yinchuan[yinchuan['node']==1].values[672:1344,-1],color='royalblue',linestyle='-',linewidth=1,label='2nd week')
plt.ylabel('Traffic speed (km/h)',font2)
plt.xlabel('Ningxia-YC',font2)
plt.xticks(xlabel, [weeks[i//96] for i in xlabel])
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')

plt.subplot(1,2,2)

xlabel = [i for i in range(0, 2016,288)]
plt.plot(metr[metr['node']==2].values[:2016,-1]*1.6,color='#ff5b00',linestyle='-',linewidth=1,label='1st week')
plt.plot(metr[metr['node']==2].values[2016:4032,-1]*1.6,color='royalblue',linestyle='-',linewidth=1, label='2nd week')
plt.xticks(xlabel, [weeks[i//288] for i in xlabel])
# plt.ylabel('Traffic speed (km/h)',font2)
plt.xlabel('METR-LA',font2)
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.show()
'''


# 用于显示两个数据集的数据分布情况
'''
plt.figure()
plt.subplot(3,1,1)
plt.plot(yinchuan[yinchuan['node']==1].values[:2000,-1],color='#ff5b00',linestyle='-',linewidth=1,label='Ningxia YC')
plt.plot(metr[metr['node']==1].values[:2000,-1]*1.6,color='royalblue',linestyle='-',linewidth=1,label='METR-LA')
plt.xticks([])
plt.ylabel('Traffic speed (km/h)',font2)
plt.legend(loc='upper left',prop=font1)
# plt.grid(axis='y')

plt.subplot(3,1,2)
plt.plot(yinchuan[yinchuan['node']==5].values[:2000,-1],color='#ff5b00',linestyle='-',linewidth=1,label='Ningxia YC')
plt.plot(metr[metr['node']==6].values[:2000,-1]*1.6,color='royalblue',linestyle='-',linewidth=1,label='METR-LA')
plt.xticks([])
plt.ylabel('Traffic speed (km/h)',font2)

plt.subplot(3,1,3)
weeks = ['2021.6.1 00:00', '2021.6.6 00:00', '2021.6.11 00:00', '2021.6.16 00:00', '2021.6.21 00:00']
xlabel = [i for i in range(0, 2001,480)]
print(xlabel)
plt.plot(yinchuan[yinchuan['node']==102].values[:2000,-1],color='#ff5b00',linestyle='-',linewidth=1,label='Ningxia YC')
plt.plot(metr[metr['node']==15].values[:2000,-1]*1.6,color='royalblue',linestyle='-',linewidth=1,label='METR-LA')
plt.xticks(xlabel, [weeks[i//480] for i in xlabel])
plt.ylabel('Traffic speed (km/h)',font2)
# plt.xlabel('Number of samples',font2)
plt.show()
'''











# STGIN

# 显示周期性
'''
weeks = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.figure()
plt.subplot(1,2,1)
xlabel = [i for i in range(0, 672,96)]
plt.plot(yinchuan[yinchuan['node']==1].values[:672,-1],color='#ff5b00',linestyle='-',linewidth=1,label='1st week')
plt.plot(yinchuan[yinchuan['node']==1].values[672:1344,-1],color='royalblue',linestyle='-',linewidth=1,label='2nd week')
plt.ylabel('Traffic speed (km/h)',font2)
# plt.xlabel('Ningxia-YC',font2)
plt.xticks(xlabel, [weeks[i//96] for i in xlabel])
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')

plt.subplot(1,2,2)
plt.plot(yinchuan[yinchuan['node']==5].values[:672,-1],color='#ff5b00',linestyle='-',linewidth=1,label='1st week')
plt.plot(yinchuan[yinchuan['node']==5].values[672:1344,-1],color='royalblue',linestyle='-',linewidth=1,label='2nd week')
plt.xticks(xlabel, [weeks[i//96] for i in xlabel])
# plt.ylabel('Traffic speed (km/h)',font2)
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.show()
'''


# 可视化每个模型在MAE，RMSE和MAPE上的一个表现
'''
ax=plt.subplot(1,3,1)
plt.plot(range(1,13,1),T_GCN_MAE_1,marker='*', color='#82cafc',linestyle='-', linewidth=1,label='T-GCN')
plt.plot(range(1,13,1),STGCN_MAE_1,marker='D',color='black',linestyle='-', linewidth=1,label='STGCN')
plt.plot(range(1,13,1),DCRNN_MAE_1,marker='P',color='red',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,13,1),AGCRN_MAE_1,marker='h',color='blue',linestyle='-', linewidth=1,label='AGCRN')
plt.plot(range(1,13,1),ASTGCN_MAE_1,marker='o',color='orange',linestyle='-', linewidth=1,label='ASTGCN')
plt.plot(range(1,13,1), Graph_WaveNet_MAE_1,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='Graph-WaveNet')
plt.plot(range(1,13,1), GMAN_MAE_1,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='GMAN')
plt.plot(range(1,13,1),ST_GRAT_MAE_1,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,13,1),MTGNN_MAE_1 ,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='MTGNN')
plt.plot(range(1,13,1), STGIN_MAE_1,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='STGIN')
ymajorFormatter = FormatStrFormatter('%1.1f')
ax.yaxis.set_major_formatter(ymajorFormatter)
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.xlabel('Target time steps',font2)
plt.ylabel('MAE',font2)
# plt.title('MAE',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,3,2)
# plt.xticks(range(1,8), range(0,31,5))
plt.plot(range(1,13,1),T_GCN_RMSE_1,marker='*', color='#82cafc',linestyle='-', linewidth=1,label='T-GCN')
plt.plot(range(1,13,1),STGCN_RMSE_1,marker='D',color='black',linestyle='-', linewidth=1,label='STGCN')
plt.plot(range(1,13,1),DCRNN_RMSE_1,marker='P',color='red',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,13,1),AGCRN_RMSE_1,marker='h',color='blue',linestyle='-', linewidth=1,label='AGCRN')
plt.plot(range(1,13,1),ASTGCN_RMSE_1,marker='o',color='orange',linestyle='-', linewidth=1,label='ASTGCN')
plt.plot(range(1,13,1), Graph_WaveNet_RMSE_1,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='Graph-WaveNet')
plt.plot(range(1,13,1), GMAN_RMSE_1,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='GMAN')
plt.plot(range(1,13,1),ST_GRAT_RMSE_1,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,13,1),MTGNN_RMSE_1,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='MTGNN')
plt.plot(range(1,13,1), STGIN_RMSE_1,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='STGIN')
# plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.xlabel('Target time steps',font2)
plt.ylabel('RMSE',font2)
# plt.title('RMSE',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,3,3)
plt.plot(range(1,13,1),T_GCN_MAPE_1,marker='*', color='#82cafc',linestyle='-', linewidth=1,label='T-GCN')
plt.plot(range(1,13,1),STGCN_MAPE_1,marker='D',color='black',linestyle='-', linewidth=1,label='STGCN')
plt.plot(range(1,13,1),DCRNN_MAPE_1,marker='P',color='red',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,13,1),AGCRN_MAPE_1,marker='h',color='blue',linestyle='-', linewidth=1,label='AGCRN')
plt.plot(range(1,13,1),ASTGCN_MAPE_1,marker='o',color='orange',linestyle='-', linewidth=1,label='ASTGCN')
plt.plot(range(1,13,1), Graph_WaveNet_MAPE_1,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='Graph-WaveNet')
plt.plot(range(1,13,1), GMAN_MAPE_1,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='GMAN')
plt.plot(range(1,13,1),ST_GRAT_MAPE_1,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,13,1),MTGNN_MAPE_1,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='MTGNN')
plt.plot(range(1,13,1), STGIN_MAPE_1,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='STGIN')
plt.ylabel('MAPE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('MAPE (%)',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.show()
'''