# -- coding: utf-8 --

import numpy as np
import pandas as pd
import argparse
from models.hyparameter import parameter
from models.utils import metric
file='/Users/guojianzou/Traffic-speed-prediction/STGIN/data/speed/train_15.csv'

class HA():
    def __init__(self, hp=None):
        self.data_divide=hp.divide_ratio           # the divide between in training set and test set ratio
        self.para=hp
        self.data=self.get_source_data(file)

        self.length=self.data.values.shape[0]  #data length

    def get_source_data(self,file_path):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        return data

    def model(self):
        self.dictionary_label = []
        self.dictionary_predict = []

        for site in range(0,108):
            data1=self.data[(self.data['node']==self.data.values[site][0])]
            for h in range(24):
                data2 = data1.loc[data1['hour'] == h]
                for minute in range(0,60,15):
                    data3 = data2.loc[data1['minute'] == minute]
                    data_value=data3.values
                    shape=data_value.shape

                    predict=np.mean(data_value[60:int(shape[0]*self.data_divide),-1])
                    label=np.reshape(data_value[int(shape[0]*self.data_divide):,-1],newshape=[-1])

                    self.dictionary_label.append(list(label))
                    self.dictionary_predict.append([predict]*label.shape[-1])

if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    ha=HA(hp=para)
    ha.model()
    metric(np.reshape(np.array(ha.dictionary_predict),newshape=[-1]),np.reshape(np.array(ha.dictionary_label),newshape=[-1]))
    # print(iter.data.loc[iter.data['ZoneID']==0])