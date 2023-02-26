# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from models.inits import *
from matplotlib.ticker import MaxNLocator



RST = pd.read_csv('RST.csv',encoding='utf-8').values[108:]

K = 1
site_num=108
RST_pre = []
RST_obs = []

for i in range(0, RST.shape[0], site_num):
    RST_obs.append(RST[i:i + site_num,37:49])
    RST_pre.append(RST[i:i + site_num, 49:])
RST_obs=np.concatenate(RST_obs,axis=-1)
RST_pre=np.concatenate(RST_pre,axis=-1)
# print(STGIN[0][[1,7,13]])
print(RST_pre.shape,RST_obs[0,:12])



font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
# plt.ylabel('Loss(ug/m3)',font2)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
plt.figure()
# k=4  # 1时候可以
h=1200
for k in range(0,108):
    plt.plot(range(1, h+1), RST_obs[k,:h], color='#0cdc73', label=u'Observed', linewidth=1)
    plt.plot(range(1, h+1), RST_pre[k,:h], color='red', label=u'3S-TAEN', linewidth=1)
    plt.legend(loc='upper left', prop=font2)
    plt.ylabel("Traffic speed", font1)
    plt.show()


