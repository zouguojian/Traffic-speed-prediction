# -- coding: utf-8 --

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

file='/Users/guojianzou/Traffic-speed-prediction/data/data_hour/'

class svm_i():
    def __init__(self,
                 site_id=0,
                 is_training=True,
                 time_size=3,
                 prediction_size=1,
                 data_divide=0.9,
                 window_step=1,
                 normalize=False):
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
        self.data=self.get_source_data(file+'train.csv')

        self.length=self.data.values.shape[0]  #data length
        self.normalize=normalize

    def get_source_data(self,file_path):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        return data

    def describe(self, label, predict):
        '''
        :param label:
        :param predict:
        :param prediction_size:
        :return:
        '''
        plt.figure()
        # Label is observed value,Blue
        plt.plot(label[0:], 'b', label=u'actual value')
        # Predict is predicted value，Red
        plt.plot(predict[0:], 'r', label=u'predicted value')
        # use the legend
        plt.legend()
        # plt.xlabel("time(hours)", fontsize=17)
        # plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
        # plt.title("the prediction of pm$_{2.5}", fontsize=17)
        plt.show()

    def accuracy(self,label,predict):
        '''
        :param Label: represents the observed value
        :param Predict: represents the predicted value
        :param epoch:
        :param steps:
        :return:
        '''
        # self.describe(label,predict)

        print(label.shape,predict.shape)
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

    def train_data(self,data,input_length,predict_length):
        low,high=0,data.shape[0]
        x ,y=[],[]
        while low+predict_length+input_length<high:
            x.append(np.reshape(data[low:low+input_length],newshape=[-1]))
            y.append(np.reshape(data[low+input_length:low+input_length+predict_length],newshape=[-1]))
            low+=1
        return np.array(x), np.array(y)

    def model(self):
        self.dictionary_label = []
        self.dictionary_predict = []

        for site in range(49):
            data1=self.data[(self.data['in_id']==self.data.values[site][0]) & (self.data['out_id']==self.data.values[site][1])]
            x = data1.values[:, -2:-1]

            x,y=self.train_data(data=x,input_length=6,predict_length=1)

            train_size = int(len(x) * 0.9)
            train_x, train_y, test_x,test_y = x[:train_size],y[:train_size], x[train_size:],y[train_size:]
            print(train_x.shape, train_y.shape, test_x.shape,test_y.shape)
            # print(data1.shape)
            model=svm.NuSVR(nu=0.457,C=.8,degree=3)

            model.fit(X=train_x, y=train_y)

            pre=model.predict(X=test_x)
            self.dictionary_predict.append(pre)
            self.dictionary_label.append(test_y)
#
if __name__=='__main__':

    ha=svm_i(site_id=0,normalize=False)

    ha.model()
    ha.accuracy(np.reshape(np.array(ha.dictionary_label),newshape=[-1]),np.reshape(np.array(ha.dictionary_predict),newshape=[-1]))
    # print(iter.data.loc[iter.data['ZoneID']==0])