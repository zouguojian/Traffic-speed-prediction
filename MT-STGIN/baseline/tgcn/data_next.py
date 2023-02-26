# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
import argparse
from baseline.tgcn.hyparameter import parameter
import pandas as pd

class DataClass(object):
    def __init__(self, hp=None):
        '''
        :param hp:
        '''
        self.hp = hp                              # hyperparameter
        self.min_value=0.000000000001
        self.input_length=hp.input_length         # time series length of input
        self.output_length=hp.output_length       # the length of prediction
        self.is_training=hp.is_training           # true or false
        self.divide_ratio=hp.divide_ratio          # the divide between in training set and test set ratio
        self.step=hp.step                         # windows step
        self.site_num=hp.site_num
        self.file_train_path= hp.file_train
        self.normalize = hp.normalize             # data normalization

        self.get_data(self.hp.file_train)
        self.length=self.data.shape[0]                        # data length
        self.get_max_min(self.data)                           # max and min values' dictionary
        self.normalization(self.data, ['speed'], hp.normalize) # normalization

    def get_data(self, file_path=None):
        '''
        :param file_path:
        :return:
        '''
        self.data = pd.read_csv(file_path, encoding='utf-8')
        self.shape=self.data.shape

    def get_max_min(self,data=None):
        '''
        :param data:
        :return:
        '''
        self.min_dict=dict()
        self.max_dict=dict()

        for key in data.keys():
            self.min_dict[key] = data[key].min()
            self.max_dict[key] = data[key].max()
        print('the max feature list is :', self.max_dict)
        print('the min feature list is :', self.min_dict)

    def normalization(self, data, keys=[], is_normalize=True):
        '''
        :param data:
        :param keys:  is a list
        :param is_normalize:
        :return:
        '''
        if is_normalize:
            for key in keys:
                data[key]=(self.data[key] - self.min_dict[key]) / (self.max_dict[key] - self.min_dict[key]+self.min_value)

    def generator(self):
        '''
        :return: yield the data of every time, input shape: [batch, site_num*(input_length+output_length)*features]
        label:   [batch, site_num, output_length]
        '''
        # 周 x 周天数 x 小时数 x 分钟段
        data = self.data.values
        if self.is_training:
            low, high = 0, int(self.shape[0]//self.site_num * self.divide_ratio)
        else:
            low, high = int(self.shape[0]//self.site_num * self.divide_ratio), int(self.shape[0]//self.site_num)

        while low + self.input_length + self.output_length <= high:
            label=data[(low + self.input_length) * self.site_num: (low + self.input_length + self.output_length) * self.site_num,-1:]
            label=np.concatenate([label[i * self.site_num : (i + 1) * self.site_num, :] for i in range(self.output_length)], axis=1)

            yield (data[low * self.site_num : (low + self.input_length) * self.site_num, 5:6],
                   label,
                   data[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 2],
                   data[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 3],
                   data[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 4]//15)
            if self.is_training:
                low += self.step
            else:
                # low += self.hp.predict_length
                low += self.output_length

    def next_batch(self, batch_size, epoch, is_training=True):
        '''
        :param batch_size:
        :param epochs:
        :param is_training:
        :return: x shape is [batch, input_length*site_num, features];
                 day shape is [batch, (input_length+output_length)*site_num];
                 hour shape is [batch, (input_length+output_length)*site_num];
                 minute shape is [batch, (input_length+output_length)*site_num];
                 label shape is [batch, output_length*site_num, features]
        '''

        self.is_training=is_training
        dataset=tf.data.Dataset.from_generator(self.generator,output_types=(tf.float32, tf.float32, tf.int32, tf.int32, tf.int32))

        if self.is_training:
            dataset=dataset.shuffle(buffer_size=int(self.shape[0]//self.hp.site_num * self.divide_ratio-self.input_length-self.output_length)//self.step)
            dataset=dataset.repeat(count=epoch)
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()

        return iterator.get_next()
# #
if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    iter=DataClass(hp=para)
    print(iter.data.keys())

    next=iter.next_batch(batch_size=12, epoch=1, is_training=False)
    with tf.Session() as sess:
        for _ in range(4):
            x, d, h, m, y=sess.run(next)
            print(x.shape)
            print(y.shape)
            print(d[0,0],h[0,0],m[0,0])