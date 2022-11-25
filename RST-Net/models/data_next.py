# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
import argparse
from models.hyparameter import parameter
import pandas as pd

class DataClass(object):
    def __init__(self, hp=None):
        '''
        :param hp:
        '''
        self.hp = hp                              # hyperparameter
        self.min_value=0.000000000001
        self.input_length=self.hp.input_length         # time series length of input
        self.output_length=self.hp.output_length       # the length of prediction
        self.is_training=self.hp.is_training           # true or false
        self.divide_ratio=self.hp.divide_ratio         # the divide between in training set and test set ratio
        self.step=self.hp.step                         # windows step
        self.site_num=self.hp.site_num
        self.file_train_s= self.hp.file_train_s
        self.file_train_p = self.hp.file_train_p
        self.normalize = self.hp.normalize             # data normalization

        self.data_s=self.get_source_data(self.file_train_s)
        self.shape_s=self.data_s.shape
        self.data_p=self.get_source_data(self.file_train_p)
        self.shape_p=self.data_p.shape

        self.length=self.data_s.shape[0]                        # data length
        self.max_s, self.min_s= self.get_max_min(self.data_s)   # max and min values' dictionary
        self.max_p, self.min_p = self.get_max_min(self.data_p)  # max and min values' dictionary

        self.normalization(self.data_s, ['speed'], max_dict=self.max_s, min_dict=self.min_s, is_normalize=self.normalize)                  # normalization
        self.normalization(self.data_p, list(self.data_p.keys())[4:], max_dict=self.max_p, min_dict=self.min_p, is_normalize=self.normalize)  # normalization

    def get_source_data(self,file_path=None):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        return data

    def get_max_min(self,data=None):
        '''
        :param data:
        :return:
        '''
        min_dict=dict()
        max_dict=dict()

        for key in data.keys():
            min_dict[key] = data[key].min()
            max_dict[key] = data[key].max()
        # print('the max feature list is :', max_dict)
        # print('the min feature list is :', min_dict)
        return max_dict, min_dict

    def normalization(self, data, keys=None, max_dict =None, min_dict=None, is_normalize=True):
        '''
        :param data:
        :param keys:  is a list
        :param is_normalize:
        :return:
        '''
        if is_normalize:
            for key in keys:
                data[key]=(data[key] - min_dict[key]) / (max_dict[key] - min_dict[key] + self.min_value)

    def generator(self):
        '''
        :return: yield the data of every time, input shape: [batch, site_num*(input_length+output_length)*features]
        label:   [batch, site_num, output_length]
        '''
        data_s = self.data_s.values
        data_p = self.data_p.values
        if self.is_training:
            low, high = 0, int(self.shape_s[0]//self.site_num * self.divide_ratio)
        else:
            low, high = int(self.shape_s[0]//self.site_num * self.divide_ratio), int(self.shape_s[0]//self.site_num)

        while low + self.input_length + self.output_length <= high:
            label=data_s[(low + self.input_length) * self.site_num: (low + self.input_length + self.output_length) * self.site_num,-1:]
            label=np.concatenate([label[i * self.site_num : (i + 1) * self.site_num, :] for i in range(self.output_length)], axis=1)

            yield (data_s[low * self.site_num : (low + self.input_length) * self.site_num, 5:6],
                   data_s[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 2],
                   data_s[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 3],
                   data_s[low * self.site_num : (low + self.input_length + self.output_length) * self.site_num, 4]//15,
                   label,
                   data_p[low:low + self.input_length, 4:],
                   data_p[low + self.input_length: low+self.input_length+self.output_length,4])
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
        dataset=tf.data.Dataset.from_generator(self.generator,output_types=(tf.float32, tf.int32, tf.int32, tf.int32, tf.float32, tf.float32, tf.float32))

        if self.is_training:
            dataset=dataset.shuffle(buffer_size=int(self.shape_s[0]//self.hp.site_num * self.divide_ratio-self.input_length-self.output_length)//self.step)
            dataset=dataset.repeat(count=epoch)
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()

        return iterator.get_next()
# #
if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    iter=DataClass(hp=para)
    print(iter.data_s.keys())

    next=iter.next_batch(batch_size=12, epoch=1, is_training=False)
    with tf.Session() as sess:
        for _ in range(4):
            x, d, h, m, y=sess.run(next)
            print(x.shape)
            print(y.shape)
            print(d[0,0],h[0,0],m[0,0])