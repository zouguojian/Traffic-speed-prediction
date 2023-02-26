# -- coding: utf-8 --
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

PSPNN_mae_1 =[4.353647,4.436225,4.534059,4.488236,4.715021,4.571190,4.516397]
PSPNN_rmse_1 =[7.324924,7.563156,7.555393,7.346906,7.786712,7.301580,7.481787]
PSPNN_mape_1 =[0.170713,0.118265,0.130777,0.114020,0.133129,0.103548,0.128409]

PSPNN_mae_2 =[5.018109,5.132795,5.258064,5.508495,5.584318,5.481236,5.330503]
PSPNN_rmse_2 =[	8.251502,8.415823,8.760375,9.008966,8.936558,8.648383,8.674441]
PSPNN_mape_2 =[0.076827,0.101001,0.070024,0.074707,0.081328,0.093152,0.082840]

PSPNN_mae_3 =[7.052771,7.004601,7.083312,7.243506,7.039853,7.323674,7.124619]
PSPNN_rmse_3 =[11.129402,11.062735,10.921598,11.247130,10.737179,11.290270,11.066346]
PSPNN_mape_3 =[0.194727,0.217832,0.206510,0.194110,0.169886,0.225496,0.201427]

MDL_mae_1 =[4.567766,4.668223,4.802578,4.801348,5.006125,4.777165,4.770534]
MDL_rmse_1 =[7.633171,7.941956,8.049773,8.048460,8.376655,7.766957,7.972980]
MDL_mape_1 =[0.179061,0.133763,0.145827,0.125346,0.143087,0.107506,0.139098]

MDL_mae_2 =[4.922186,5.050283,5.199652,5.399734,5.446237,5.300889,5.219830]
MDL_rmse_2 =[8.098513,8.267899,8.640902,8.907971,8.762857,8.452044,8.526292]
MDL_mape_2 =[0.075204,0.097207,0.069290,0.073145,0.079649,0.089566,0.080677]

MDL_mae_3 =[7.115840,7.157158,7.221766,7.438228,7.307103,7.557099,7.299532]
MDL_rmse_3 =[11.168698,11.162749,11.012158,11.347979,10.966957,11.412749,11.179714]
MDL_mape_3 =[0.191805,0.220924,0.201937,0.187723,0.168965,0.218110,0.198244]

T_GCN_mae_1 =[4.867914,4.915519,4.991863,4.900115,5.071895,4.880197,4.937917]
T_GCN_rmse_1 =[8.007648,8.129066,8.126287,7.989766,8.363378,7.866967,8.081999]
T_GCN_mape_1 =[0.183295,0.130065,0.138732,0.119302,0.142765,0.108532,0.137115]

T_GCN_mae_2 =[4.881372,4.909656,4.990881,5.137515,5.178477,5.040731,5.023106]
T_GCN_rmse_2 =[8.011009,8.144530,8.430680,8.608794,8.550889,8.144875,8.318192]
T_GCN_mape_2 =[0.074088,0.096569,0.066887,0.069687,0.075928,0.086002,0.078194]

T_GCN_mae_3 =[7.292850,7.264247,7.260326,7.410805,7.263291,7.412676,7.317366]
T_GCN_rmse_3 =[11.339722,11.330866,11.029550,11.286448,10.923553,11.250577,11.194583]
T_GCN_mape_3 =[0.192573,0.218989,0.201144,0.184970,0.164272,0.216086,0.196339]

DCRNN_mae_1 =[4.730402,4.711142,4.700361,4.727126,4.761065,4.794417,4.737419]
DCRNN_rmse_1 =[7.731863,7.739123,7.725201,7.759669,7.777472,7.858567,7.765448]
DCRNN_mape_1 =[0.131417,0.130804,0.133867,0.135671,0.134084,0.136295,0.133690]

DCRNN_mae_2 =[5.218989,5.141128,5.105221,5.100863,5.099192,5.116236,5.130271]
DCRNN_rmse_2 =[8.667619,8.588108,8.559472,8.542141,8.536303,8.546819,8.573530]
DCRNN_mape_2 =[0.082794,0.081971,0.081556,0.080885,0.080372,0.080445,0.081337]

DCRNN_mae_3 =[7.253874,7.323273,7.389776,7.408976,7.444117,7.467514,7.381255]
DCRNN_rmse_3 =[11.236532,11.301529,11.361451,11.427432,11.435319,11.478284,11.373731]
DCRNN_mape_3 =[0.226070,0.215237,0.212807,0.214020,0.212680,0.215993,0.216135]

AST_GAT_mae_1 =[4.497184,4.613954,4.649611,4.607247,4.772495,4.547935,4.614738]
AST_GAT_rmse_1 =[7.535124,7.784701,7.653306,7.662314,7.960739,7.460128,7.677789]
AST_GAT_mape_1 =[0.168395,0.120126,0.129663,0.110013,0.129681,0.100302,0.126363]

AST_GAT_mae_2 =[4.679883,4.652782,4.718185,4.913333,4.889099,4.672708,4.754332]
AST_GAT_rmse_2 =[7.822722,7.935115,8.138360,8.359120,8.272555,7.832850,8.062853]
AST_GAT_mape_2 =[0.070963,0.093704,0.063064,0.066619,0.072517,0.082804,0.074945]

AST_GAT_mae_3 =[6.892128,6.818961,6.961788,6.997957,6.797758,7.006877,6.912578]
AST_GAT_rmse_3 =[10.922516,10.779595,10.728664,10.848628,10.346283,10.841717,10.746211]
AST_GAT_mape_3 =[0.201657,0.221000,0.211088,0.187143,0.171407,0.221291,0.202264]

GMAN_mae_1 =[4.334882,4.313689,4.360220,4.222983,4.436746,4.213836,4.313725]
GMAN_rmse_1 =[7.332422,7.419843,7.373770,7.132550,7.515837,7.016173,7.300442]
GMAN_mape_1 =[0.172884,0.114356,0.129397,0.111522,0.130163,0.095924,0.125708]

GMAN_mae_2 =[4.740033,4.676142,4.746842,4.874263,4.917791,4.727566,4.780439]
GMAN_rmse_2 =[7.906706,8.000947,8.240516,8.415353,8.332593,7.890183,8.133700]
GMAN_mape_2 =[0.072855,0.095317,0.063751,0.066442,0.072201,0.081023,0.075265]

GMAN_mae_3 =[6.999796,6.907157,6.978471,7.049780,6.833681,7.030167,6.966509]
GMAN_rmse_3 =[11.002648,10.892133,10.725927,10.940084,10.504736,10.907470,10.830129]
GMAN_mape_3 =[0.204040,0.216095,0.206211,0.186343,0.165788,0.218526,0.199500]

ST_GRAT_mae_1 =[4.234032,4.306022,4.393604,4.495383,4.588520,4.675550,4.448852]
ST_GRAT_rmse_1 =[7.280299,7.386465,7.502555,7.658772,7.788842,7.913568,7.591644]
ST_GRAT_mape_1 =[0.124619,0.127905,0.131194,0.132167,0.135779,0.135204,0.131145]

ST_GRAT_mae_2 =[4.772299,4.871571,4.995002,5.165320,5.325081,5.484719,5.102332]
ST_GRAT_rmse_2 =[8.264096,8.362779,8.491458,8.677082,8.858995,9.038960,8.619865]
ST_GRAT_mape_2 =[0.076821,0.078492,0.080265,0.082695,0.084784,0.086787,0.081640]

ST_GRAT_mae_3 =[6.771106,6.891242,6.995427,7.106168,7.192948,7.274607,7.038582]
ST_GRAT_rmse_3 =[10.776956,10.918130,11.055330,11.224825,11.347829,11.465987,11.134088]
ST_GRAT_mape_3 =[0.185265,0.189862,0.193490,0.197077,0.200330,0.201765,0.194631]

MT_STGIN_mae_1 =[4.167658,4.152724,4.225301,4.129288,4.343058,4.132023,4.191676]
MT_STGIN_rmse_1 =[7.233356,7.331061,7.390471,7.154604,7.542089,6.995177,7.276548]
MT_STGIN_mape_1 =[0.161414,0.109230,0.118732,0.107747,0.125871,0.093290,0.119381]

MT_STGIN_mae_2 =[4.682971,4.588183,4.672132,4.860928,4.832805,4.613521,4.708423]
MT_STGIN_rmse_2 =[7.956369,8.014418,8.377290,8.592709,8.402509,7.933352,8.216736]
MT_STGIN_mape_2 =[0.072753,0.096213,0.064088,0.067078,0.072708,0.082680,0.075920]

MT_STGIN_mae_3 =[6.884369,6.772877,6.750407,6.943268,6.745666,6.883471,6.830010]
MT_STGIN_rmse_3 =[10.983397,10.887800,10.592183,10.859518,10.464038,10.828980,10.770837]
MT_STGIN_mape_3 =[0.183126,0.204971,0.188529,0.169706,0.149622,0.202187,0.183024]

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 17,
}
# plt.ylabel('Loss(ug/m3)',font2)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}


DCRNN = pd.read_csv('results/DCRNN.csv',encoding='utf-8').values[108*19:]
# Bi_LSTM = pd.read_csv('results/BILSTM.csv',encoding='utf-8').values[108:]
# FI_RNNs = pd.read_csv('results/FI-RNN.csv',encoding='utf-8').values[108:]
GMAN = pd.read_csv('results/GMAN.csv',encoding='utf-8').values[108:]
MT_STGIN = pd.read_csv('results/MT-STGIN.csv',encoding='utf-8').values[108:]
# PSPNN = pd.read_csv('results/PSPNN.csv',encoding='utf-8').values[108:]
ST_GRAT = pd.read_csv('results/ST-GRAT.csv',encoding='utf-8').values[108*5:]
AST_GAT=pd.read_csv('results/ASTGAT.csv',encoding='utf-8').values[108:]
T_GCN=pd.read_csv('results/T-GCN.csv',encoding='utf-8').values[108:]

DCRNN_pre = []
DCRNN_obs = []
Bi_LSTM_pre = []
Bi_LSTM_obs = []
FI_RNNs_pre = []
FI_RNNs_obs = []
GMAN_pre = []
GMAN_obs = []
MT_STGIN_pre = []
MT_STGIN_obs = []
PSPNN_pre = []
PSPNN_obs = []
ST_GRAT_pre = []
ST_GRAT_obs = []
AST_GAT_pre = []
AST_GAT_obs = []
T_GCN_pre = []
T_GCN_obs = []

K = 10
site_num=108
for i in range(0,DCRNN.shape[0],site_num * 6):
    DCRNN_obs.append(DCRNN[i:i+site_num,1:7])
    DCRNN_pre.append(DCRNN[i:i + site_num, 7:])
DCRNN_obs=np.concatenate(DCRNN_obs,axis=-1)[:,:-6]
DCRNN_pre=np.concatenate(DCRNN_pre,axis=-1)[:,:-6]
print(DCRNN_pre.shape, DCRNN_obs[0,:6])

# for i in range(site_num,site_num*K,site_num):
#     Bi_LSTM_obs.append(Bi_LSTM[i:i+site_num,19:25])
#     Bi_LSTM_pre.append(Bi_LSTM[i:i + site_num, 25:])
#
# for i in range(site_num,site_num*K,site_num):
#     FI_RNNs_obs.append(FI_RNNs[i:i+site_num,19:25])
#     FI_RNNs_pre.append(FI_RNNs[i:i + site_num, 25:])
#
for i in range(0,GMAN.shape[0],site_num):
    GMAN_obs.append(GMAN[i:i+site_num,19:25])
    GMAN_pre.append(GMAN[i:i + site_num, 25:])
GMAN_obs=np.concatenate(GMAN_obs,axis=-1)
GMAN_pre=np.concatenate(GMAN_pre,axis=-1)
print(GMAN_pre.shape, GMAN_obs[0,:6])

time =[]
for i in range(0, MT_STGIN.shape[0], site_num):
    MT_STGIN_obs.append(MT_STGIN[i:i + site_num,19:25])
    MT_STGIN_pre.append(MT_STGIN[i:i + site_num, 25:])
    time.append(list(MT_STGIN[i, :19]))
MT_STGIN_obs=np.concatenate(MT_STGIN_obs,axis=-1)
MT_STGIN_pre=np.concatenate(MT_STGIN_pre,axis=-1)
print(MT_STGIN[0][[1,7,13]])
print(MT_STGIN[108*100][[1,7,13]])
print(MT_STGIN_pre.shape,MT_STGIN_obs[0,:6])
time=np.array(time)
#
# for i in range(site_num,site_num*K,site_num):
#     PSPNN_obs.append(PSPNN[i:i+site_num,19:25])
#     PSPNN_pre.append(PSPNN[i:i + site_num, 25:])

for i in range(0,ST_GRAT.shape[0],site_num * 6):
    ST_GRAT_obs.append(ST_GRAT[i:i+site_num,1:7])
    ST_GRAT_pre.append(ST_GRAT[i:i + site_num, 7:])
ST_GRAT_obs=np.concatenate(ST_GRAT_obs,axis=-1)[:,:-6]
ST_GRAT_pre=np.concatenate(ST_GRAT_pre,axis=-1)[:,:-6]
print(ST_GRAT_pre.shape, ST_GRAT_obs[0,:6])


for i in range(0,AST_GAT.shape[0],site_num):
    AST_GAT_obs.append(AST_GAT[i:i+site_num,19:25])
    AST_GAT_pre.append(AST_GAT[i:i + site_num, 25:])
AST_GAT_obs=np.concatenate(AST_GAT_obs,axis=-1)
AST_GAT_pre=np.concatenate(AST_GAT_pre,axis=-1)
print(AST_GAT_pre.shape,AST_GAT_obs[0,:6])

for i in range(0,T_GCN.shape[0],site_num):
    T_GCN_obs.append(T_GCN[i:i+site_num,19:25])
    T_GCN_pre.append(T_GCN[i:i + site_num, 25:])
T_GCN_obs=np.concatenate(T_GCN_obs,axis=-1)
T_GCN_pre=np.concatenate(T_GCN_pre,axis=-1)
print(T_GCN_pre.shape,T_GCN_obs[0,:6])

# print(time[0])
# print(time[17])
# print(time[34])
# print(time[50])
# print(time[67])
# print(time[84])
# print(time[100])


# '''
plt.subplot(3, 1, 1)
print(MT_STGIN_pre.shape)
k=1
# print(MT_STGIN_obs[i][j])
# plt.xticks(['2021.8.14 6:45', '7:00', '7:15', '7:30', '7:45', '8:00'])
# plt.xticks(range(1, 7), ['2021.8.14 6:45', '7:00', '7:15', '7:30', '7:45', '8:00'])
plt.plot(range(1, 600+1), MT_STGIN_obs[k,:600], color='blue', label=u'Observed', linewidth=1)
plt.plot(range(1, 600+1), MT_STGIN_pre[k,:600], color='#a55af4', label=u'MT-STGIN', linewidth=1)
plt.plot(range(1, 600+1), DCRNN_pre[k,:600], color='#f504c9', label=u'DCRNN', linewidth=1)
plt.plot(range(1, 600+1), GMAN_pre[k,:600], color='#d0c101', label=u'GMAN', linewidth=1)
plt.plot(range(1, 600+1), AST_GAT_pre[k,:600], color='#82cafc', label=u'AST-GAT', linewidth=1)
# plt.plot(range(1, 7), PSPNN_pre[i][j], marker='s', color='#0cdc73', label=u'PSPNN', linewidth=1)
# plt.plot(range(1, 7), T_GCN_pre[i][j], marker='d', color='#ff5b00', label=u'T-GCN', linewidth=1)
plt.legend(loc='upper left', prop=font2)
# plt.xlabel("Target time steps", font2)
plt.ylabel("Traffic speed", font1)
# plt.title("Road segment 1", font1)

# i,j=8,10
# print(MT_STGIN_obs[i][j])
plt.subplot(3, 1, 2)
k=69
# plt.xticks(range(1, 7), ['2021.8.14 6:45', '7:00', '7:15', '7:30', '7:45', '8:00'])
plt.plot(range(1, 600+1), MT_STGIN_obs[k,:600], color='blue', label=u'Observed', linewidth=1)
plt.plot(range(1, 600+1), MT_STGIN_pre[k,:600], color='#a55af4', label=u'MT-STGIN', linewidth=1)
plt.plot(range(1, 600+1), DCRNN_pre[k,:600], color='#f504c9', label=u'DCRNN', linewidth=1)
plt.plot(range(1, 600+1), GMAN_pre[k,:600], color='#d0c101', label=u'GMAN', linewidth=1)
# plt.plot(range(1, 7), PSPNN_pre[i][j], marker='s', color='#0cdc73', label=u'PSPNN', linewidth=1)
# plt.plot(range(1, 7), T_GCN_pre[i][j], marker='d', color='#ff5b00', label=u'T-GCN', linewidth=1)
plt.plot(range(1, 600+1), AST_GAT_pre[k,:600], color='#82cafc', label=u'AST-GAT', linewidth=1)
# plt.legend(loc='upper left', prop=font2)
# plt.xlabel("Target time steps", font2)
plt.ylabel("Traffic speed", font1)
# plt.title("Road segment 2", font1)

# i,j=8,97
# print(MT_STGIN_obs[i][j])
plt.subplot(3, 1, 3)
k=92
# plt.xticks(range(1, 7), ['2021.8.14 6:45', '7:00', '7:15', '7:30', '7:45', '8:00'])
plt.plot(range(1, 600+1), MT_STGIN_obs[k,:600], color='blue', label=u'Observed', linewidth=1)
plt.plot(range(1, 600+1), MT_STGIN_pre[k,:600], color='#a55af4', label=u'MT-STGIN', linewidth=1)
plt.plot(range(1, 600+1), DCRNN_pre[k,:600], color='#f504c9', label=u'DCRNN', linewidth=1)
plt.plot(range(1, 600+1), GMAN_pre[k,:600], color='#d0c101', label=u'GMAN', linewidth=1)
plt.plot(range(1, 600+1), AST_GAT_pre[k,:600], color='#82cafc', label=u'AST-GAT', linewidth=1)
# plt.plot(range(1, 7), PSPNN_pre[i][j], marker='s', color='#0cdc73', label=u'PSPNN', linewidth=1)
# plt.plot(range(1, 7), T_GCN_pre[i][j], marker='d', color='#ff5b00', label=u'T-GCN', linewidth=1)
# plt.legend(loc='upper left', prop=font2)
# plt.xlabel("Target time steps", font2)
plt.ylabel("Traffic speed", font1)
# plt.title("Road segment 3", font1)

plt.show()
# '''


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


'''
i,j=0,1000
l,h=0,28
# y=x的拟合可视化图
# plt.figure()

# plt.subplot(2,3,1)
# plt.scatter(DCRNN_obs[l:h,i:j],DCRNN_pre[l:h,i:j],alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'DCRNN',linewidths=1)
a=[i for i in range(150)]
b=[i for i in range(150)]
# plt.plot(a,b,'black',linewidth=2)
# #plt.scatter(a, b)
# # plt.title("Entrance tall dataset", font2)
# #设置横纵坐标的名称以及对应字体格式
# plt.ylabel("Predicted traffic speed", font2)
# plt.legend(loc='upper left',prop=font2)

# plt.subplot(2,3,2)
# plt.scatter(ST_GRAT_obs[l:h,i:j],ST_GRAT_pre[l:h,i:j],alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'ST-GRAT',linewidths=1)
# plt.plot(a,b,'black',linewidth=2)
# #plt.scatter(a, b)
# # plt.title("Exit tall dataset", font2)
# plt.legend(loc='upper left',prop=font2)

# plt.subplot(2,3,3)
# plt.scatter(T_GCN_obs[l:h,i:j],T_GCN_pre[l:h,i:j],alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'T-GCN',linewidths=1)
c=[i for i in range(150)]
d=[i for i in range(150)]
# plt.plot(c,d,'black',linewidth=2)
# # plt.title("Gantry dataset", font2)
# #设置横纵坐标的名称以及对应字体格式
# plt.legend(loc='upper left',prop=font2)

plt.subplot(1,3,1)
plt.scatter(AST_GAT_obs[l:h,i:j],AST_GAT_pre[l:h,i:j],alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'AST-GAT',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic speed", font2)
plt.ylabel("Predicted traffic speed", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(1,3,2)
plt.scatter(GMAN_obs[l:h,i:j],GMAN_pre[l:h,i:j],alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'GMAN',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic speed", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(1,3,3)
plt.scatter(MT_STGIN_obs[l:h,i:j],MT_STGIN_pre[l:h,i:j],alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'MT-STGIN',linewidths=1)
plt.plot(c,d,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic speed", font2)
plt.legend(loc='upper left',prop=font2)
plt.show()
'''


# 可视化每个模型在MAPE上的一个表现，柱状图
'''
x = np.arange(1, 9, 1)
total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2
plt.subplot(1,2,1)

rmse_1=[DCRNN_rmse_1[-1],ST_GRAT_rmse_1[-1],PSPNN_rmse_1[-1],MDL_rmse_1[-1],GMAN_rmse_1[-1],T_GCN_rmse_1[-1],AST_GAT_rmse_1[-1],MT_STGIN_rmse_1[-1]]
rmse_2=[DCRNN_rmse_2[-1],ST_GRAT_rmse_2[-1],PSPNN_rmse_2[-1],MDL_rmse_2[-1],GMAN_rmse_2[-1],T_GCN_rmse_2[-1],AST_GAT_rmse_2[-1],MT_STGIN_rmse_2[-1]]
rmse_3=[DCRNN_rmse_3[-1],ST_GRAT_rmse_3[-1],PSPNN_rmse_3[-1],MDL_rmse_3[-1],GMAN_rmse_3[-1],T_GCN_rmse_3[-1],AST_GAT_rmse_3[-1],MT_STGIN_rmse_3[-1]]
mape_1=[DCRNN_mape_1[-1],ST_GRAT_mape_1[-1],PSPNN_mape_1[-1],MDL_mape_1[-1],GMAN_mape_1[-1],T_GCN_mape_1[-1],AST_GAT_mape_1[-1],MT_STGIN_mape_1[-1]]
mape_2=[DCRNN_mape_2[-1],ST_GRAT_mape_2[-1],PSPNN_mape_2[-1],MDL_mape_2[-1],GMAN_mape_2[-1],T_GCN_mape_2[-1],AST_GAT_mape_2[-1],MT_STGIN_mape_2[-1]]
mape_3=[DCRNN_mape_3[-1],ST_GRAT_mape_3[-1],PSPNN_mape_3[-1],MDL_mape_3[-1],GMAN_mape_3[-1],T_GCN_mape_3[-1],AST_GAT_mape_3[-1],MT_STGIN_mape_3[-1]]
plt.ylim(5,12)
plt.xticks(range(1,9),['DCRNN','ST-GRAT','PSPNN','MDL','GMAN','T-GCN','AST-GAT','MT-STGIN'])
plt.bar(x, rmse_1, width=width,label='ETTG',color = 'red')
plt.bar(x + width, rmse_2, width=width,label='GTG',color = 'black')
plt.bar(x + 2 * width, rmse_3, width=width,label='GTET',color='salmon')
plt.ylabel('RMSE',font2)
# plt.title('Target time steps $Q$ = 6 ([0-30 min])',font2)
plt.legend()

plt.subplot(1,2,2)
plt.ylim(0.0, 0.25)
plt.xticks(range(1,9),['DCRNN','ST-GRAT','PSPNN','MDL','GMAN','T-GCN','AST-GAT','MT-STGIN'])
plt.bar(x, mape_1, width=width,label='ETTG',color = 'red')
plt.bar(x + width, mape_2, width=width,label='GTG',color = 'black')
plt.bar(x + 2 * width, mape_3, width=width,label='GTET',color='salmon')
plt.ylabel('MAPE',font2)
# plt.title('Target time steps $Q$ = 6 ([0-30 min])',font2)
plt.legend()
plt.show()
'''


# 可视化每个模型在MAE，RMSE和MAPE上的一个表现
'''
plt.figure()
plt.subplot(1,3,1)
# plt.plot(range(1,7,1),LSTM_mae,marker='P',color='red',linestyle='-', linewidth=1,label='LSTM')
plt.plot(range(1,7,1),DCRNN_mae_3[:6],marker='h',color='blue',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,7,1),ST_GRAT_mae_3[:6],marker='o',color='orange',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,7,1), PSPNN_mae_3[:6],marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='PSPNN')
plt.plot(range(1,7,1), MDL_mae_3[:6],marker='p', color='#f504c9',linestyle='-',linewidth=1,label='MDL')
plt.plot(range(1,7,1),GMAN_mae_3[:6],marker='^',color='#d0c101',linestyle='-', linewidth=1,label='GMAN')
plt.plot(range(1,7,1),T_GCN_mae_3[:6],marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='T-GCN')
plt.plot(range(1,7,1), AST_GAT_mae_3[:6],marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,7,1), MT_STGIN_mae_3[:6],marker='X', color='#a55af4',linestyle='-',linewidth=1,label='MT-STGIN')
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.ylabel('MAE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('MAE',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,3,2)
# plt.xticks(range(1,8), range(0,31,5))
# plt.plot(range(1,7,1),LSTM_rmse,marker='P',color='red',linestyle='-', linewidth=1,label='LSTM')
plt.plot(range(1,7,1),DCRNN_rmse_3[:6],marker='h',color='blue',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,7,1),ST_GRAT_rmse_3[:6],marker='o',color='orange',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,7,1), PSPNN_rmse_3[:6],marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='PSPNN')
plt.plot(range(1,7,1), MDL_rmse_3[:6],marker='p', color='#f504c9',linestyle='-',linewidth=1,label='MDL')
plt.plot(range(1,7,1),GMAN_rmse_3[:6],marker='^',color='#d0c101',linestyle='-', linewidth=1,label='GMAN')
plt.plot(range(1,7,1),T_GCN_rmse_3[:6],marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='T-GCN')
plt.plot(range(1,7,1), AST_GAT_rmse_3[:6],marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,7,1), MT_STGIN_rmse_3[:6],marker='X', color='#a55af4',linestyle='-',linewidth=1,label='MT-STGIN')
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.ylabel('RMSE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,3,3)
# plt.plot(range(1,7,1),LSTM_mape,marker='P',color='red',linestyle='-', linewidth=1,label='LSTM')
plt.plot(range(1,7,1),DCRNN_mape_3[:6],marker='h',color='blue',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,7,1),ST_GRAT_mape_3[:6],marker='o',color='orange',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,7,1), PSPNN_mape_3[:6],marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='PSPNN')
plt.plot(range(1,7,1), MDL_mape_3[:6],marker='p', color='#f504c9',linestyle='-',linewidth=1,label='MDL')
plt.plot(range(1,7,1),GMAN_mape_3[:6],marker='^',color='#d0c101',linestyle='-', linewidth=1,label='GMAN')
plt.plot(range(1,7,1),T_GCN_mape_3[:6],marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='T-GCN')
plt.plot(range(1,7,1), AST_GAT_mape_3[:6],marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,7,1), MT_STGIN_mape_3[:6],marker='X', color='#a55af4',linestyle='-',linewidth=1,label='MT-STGIN')
plt.ylabel('MAPE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('Gantry dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.show()
'''

# 可视化不同数据集上的 MAE
'''
plt.subplot(1,3,1)
# plt.plot(range(1,7,1),LSTM_mae,marker='P',color='red',linestyle='-', linewidth=1,label='LSTM')
plt.plot(range(1,7,1),DCRNN_mae_1[:6],marker='h',color='blue',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,7,1),ST_GRAT_mae_1[:6],marker='o',color='orange',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,7,1), PSPNN_mae_1[:6],marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='PSPNN')
plt.plot(range(1,7,1), MDL_mae_1[:6],marker='p', color='#f504c9',linestyle='-',linewidth=1,label='MDL')
plt.plot(range(1,7,1),GMAN_mae_1[:6],marker='^',color='#d0c101',linestyle='-', linewidth=1,label='GMAN')
plt.plot(range(1,7,1),T_GCN_mae_1[:6],marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='T-GCN')
plt.plot(range(1,7,1), AST_GAT_mae_1[:6],marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,7,1), MT_STGIN_mae_1[:6],marker='X', color='#a55af4',linestyle='-',linewidth=1,label='MT-STGIN')
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.ylabel('MAE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('MAE',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,3,2)
# plt.xticks(range(1,8), range(0,31,5))
# plt.plot(range(1,7,1),LSTM_rmse,marker='P',color='red',linestyle='-', linewidth=1,label='LSTM')
plt.plot(range(1,7,1),DCRNN_mae_2[:6],marker='h',color='blue',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,7,1),ST_GRAT_mae_2[:6],marker='o',color='orange',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,7,1), PSPNN_mae_2[:6],marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='PSPNN')
plt.plot(range(1,7,1), MDL_mae_2[:6],marker='p', color='#f504c9',linestyle='-',linewidth=1,label='MDL')
plt.plot(range(1,7,1),GMAN_mae_2[:6],marker='^',color='#d0c101',linestyle='-', linewidth=1,label='GMAN')
plt.plot(range(1,7,1),T_GCN_mae_2[:6],marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='T-GCN')
plt.plot(range(1,7,1), AST_GAT_mae_2[:6],marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,7,1), MT_STGIN_mae_2[:6],marker='X', color='#a55af4',linestyle='-',linewidth=1,label='MT-STGIN')
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
# plt.ylabel('RMSE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,3,3)
# plt.plot(range(1,7,1),LSTM_mape,marker='P',color='red',linestyle='-', linewidth=1,label='LSTM')
plt.plot(range(1,7,1),DCRNN_mae_3[:6],marker='h',color='blue',linestyle='-', linewidth=1,label='DCRNN')
plt.plot(range(1,7,1),ST_GRAT_mae_3[:6],marker='o',color='orange',linestyle='-', linewidth=1,label='ST-GRAT')
plt.plot(range(1,7,1), PSPNN_mae_3[:6],marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='PSPNN')
plt.plot(range(1,7,1), MDL_mae_3[:6],marker='p', color='#f504c9',linestyle='-',linewidth=1,label='MDL')
plt.plot(range(1,7,1),GMAN_mae_3[:6],marker='^',color='#d0c101',linestyle='-', linewidth=1,label='GMAN')
plt.plot(range(1,7,1),T_GCN_mae_3[:6],marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='T-GCN')
plt.plot(range(1,7,1), AST_GAT_mae_3[:6],marker='*', color='#82cafc',linestyle='-',linewidth=1,label='AST-GAT')
plt.plot(range(1,7,1), MT_STGIN_mae_3[:6],marker='X', color='#a55af4',linestyle='-',linewidth=1,label='MT-STGIN')
# plt.ylabel('MAPE',font2)
plt.xlabel('Target time steps',font2)
# plt.title('Gantry dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.show()
'''