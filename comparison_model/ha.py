# -- coding: utf-8 --

import tensorflow as tf
import numpy as np
from data.data_hour import *
import pandas as pd
import argparse
from gcn_model.hyparameter import parameter

file='/Users/guojianzou/Traffic-speed-prediction/data/data_hour/'

class HA():
    def __init__(self,
                 site_id=0,
                 is_training=True,
                 time_size=3,
                 prediction_size=1,
                 data_divide=0.9,
                 window_step=1,
                 normalize=False,
                 hp=None):
        '''
        :param is_training: while is_training is True,the model is training state
        :param field_len:
        :param time_size:
        :param prediction_size:
        :param target_site:
        '''
        self.site_id=site_id                   # ozone ID
        self.time_size=time_size               # time series length of input
        self.prediction_size=prediction_size   # the length of prediction
        self.is_training=is_training           # true or false
        self.data_divide=data_divide           # the divide between in training set and test set ratio
        self.window_step=window_step           # windows step
        self.para=hp
        self.data=self.get_source_data(file+'train.csv')

        self.length=self.data.values.shape[0]  #data length
        self.normalize=normalize

    def get_source_data(self,file_path):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        return data

    def accuracy(self,label,predict):
        '''
        :param Label: represents the observed value
        :param Predict: represents the predicted value
        :param epoch:
        :param steps:
        :return:
        '''
        error = label - predict
        average_error = np.mean(np.fabs(error.astype(float)))
        print("mae is : %.6f" % (average_error))

        rmse_error = np.sqrt(np.mean(np.square(label - predict)))
        print("rmse is : %.6f" % (rmse_error))

        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (predict - np.mean(predict)))) / (np.std(predict) * np.std(label))
        print('correlation coefficient is: %.6f' % (cor))

        sse = np.sum((label - predict) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        R2 = 1 - sse / sst
        print('r^2 is: %.6f' % (R2))

        return average_error,rmse_error,cor,R2

    def model(self):
        self.dictionary_label = []
        self.dictionary_predict = []

        for site in range(self.para.site_num):
            data1=self.data[(self.data['in_id']==self.data.values[site][0]) & (self.data['out_id']==self.data.values[site][1])]
            # print(data1.shape)
            for h in range(24):
                data2=data1.loc[data1['hour']==h]
                print(data2.shape)
                label=np.mean(data2.values[ : 110,-2])
                # predict = np.mean(data3.values[25:26, -1])
                predict=np.reshape(data2.values[110: 123,-2],newshape=[-1])
                # print(predict,predict.shape[-1])
                self.dictionary_label.append([label]*predict.shape[-1])
                # self.dictionary_label.append(label)
                # self.dictionary_predict.append(predict)
                self.dictionary_predict.append(list(predict))

#
if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    ha=HA(site_id=0,normalize=False,hp=para)
    print(ha.data.keys())
    print(ha.data)
    ha.model()
    ha.accuracy(np.reshape(np.array(ha.dictionary_label),newshape=[-1]),np.reshape(np.array(ha.dictionary_predict),newshape=[-1]))
    # print(iter.data.loc[iter.data['ZoneID']==0])