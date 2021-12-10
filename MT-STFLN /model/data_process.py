# -- coding: utf-8 --

import tensorflow as tf
import pandas as pd
import numpy as np
import csv

file_path=r'/Users/guojianzou/PycharmProjects/OD/data/Order_all.csv'
save_path=r'/Users/guojianzou/PycharmProjects/OD/data/data_all.csv'

train_path=r'/Users/guojianzou/PycharmProjects/OD/data/train_data.csv'
combine_path=r'/Users/guojianzou/PycharmProjects/OD/data/combine_data.csv'
data_colum=["ZoneID","Area","Slon","Slat","Elon","Elat","day","hour","min","second"]

def data_save(file_path,save_pave):
    '''
    :param file_name:
    :return:
    dtype pd.datafrme
    '''

    data = pd.read_csv(file_path, encoding='utf-8')
    data=data.values

    file = open(save_path, 'w', encoding='utf-8')
    writer=csv.writer(file)
    writer.writerow(data_colum)

    for line in data:
        # line = char.split(',')
        data_line=[int(line[1])]+[float(ch) for ch in line[2:7]]+[int(line[11]),int(line[12])]+[int(line[9][14:16]),int(line[9][17:19])]
        writer.writerow(data_line)
    file.close()
    print('data_save finish')


def train_data(save_path,train_path):
    train_colum = ["ZoneID", "day", "hour", "min","label"]
    file = open(train_path, 'w', encoding='utf-8')
    writer=csv.writer(file)
    writer.writerow(train_colum)

    data = pd.read_csv(save_path, encoding='utf-8')
    for d in range(1,31):
        data1=data.loc[data['day'] == d]
        if data1.values.shape[0]==0:
            print('day empty')
            continue
        for h in range(0,24):
            data2 = data1.loc[data1['hour'] == h]
            if data2.values.shape[0] == 0:
                print('hour empty')
                continue
            for m in range(0,60):
                data3 = data2.loc[data2['min'] == m]
                if data3.values.shape[0] == 0:
                    print('min empty')
                    continue
                for id in range(162):
                    data4 = data3.loc[data3['ZoneID'] == id]
                    if data4.values.shape[0] == 0:
                        print('zone empty')
                        continue
                    line=[id,d,h,m,data4.values.shape[0]]
                    writer.writerow(line)
    file.close()
    print('train_data finish!!!!')
    return


def data_combine(train_path, combine_path):
    train_colum = ["ZoneID", "day", "hour", "min-15","label"]
    file = open(combine_path, 'w', encoding='utf-8')
    writer=csv.writer(file)
    writer.writerow(train_colum)

    data = pd.read_csv(train_path, encoding='utf-8')

    for d in range(1,31):
        data1=data.loc[data['day'] == d]
        if data1.values.shape[0]==0:
            print(d,' day empty')
            continue
        for h in range(0,24):
            data2 = data1.loc[data1['hour'] == h]
            if data2.values.shape[0] == 0:
                print(d,h,' hour empty')
                continue
            for i in range(4):
                for id in range(162):
                    data3 = data2.loc[data2['ZoneID'] == id]
                    if data3.values.shape[0] == 0:
                        print(d, h, (i + 1) * 15, id,' zone empty')
                        line = [id, d, h, (i + 1) * 15, 0]
                        writer.writerow(line)
                        continue
                    sum_ = sum([data3.loc[(data['min'] == (j + i * 15))].values.shape[0] for j in range(10)])
                    line=[id,d,h,(i+1)*15,sum_]
                    writer.writerow(line)
    file.close()
    print('data_combine finish!!!!')
    return

# data=data_save(file_path,save_path)
#
# train_data(save_path,train_path)
#
# data_combine(train_path, combine_path)

def sudden_changed(city_dictionary_):
    '''
    用于处理突变的值
    Args:
        city_dictionary:
    Returns:
    '''
    if city_dictionary_:
        for key in city_dictionary_.keys():
            dataFrame=city_dictionary_[key].values
            shape=city_dictionary_[key].shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if i!=0:
                        if dataFrame[i][j]-dataFrame[i-1][j]>200:
                            dataFrame[i][j] = dataFrame[i - 1][j]
            city_dictionary_[key]=pd.DataFrame(dataFrame)
    return city_dictionary_

class DataIterator():             #切记这里的训练时段和测试时段的所用的对象不变，否则需要重复加载数据
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
        self.window_step=window_step # windows step
        self.source_data=self.get_source_data(combine_path)

        self.data=self.source_data.loc[self.source_data['ZoneID']==self.site_id]
        self.length=self.data.values.shape[0]  #data length
        self.max,self.min=self.get_max_min()   # max and min are list type, used for the later normalization
        self.normalize=normalize
        if self.normalize:self.normalization() #normalization

    def get_source_data(self,file_path):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        return data

    def get_max_min(self):
        '''
        :return: the max and min value of input features
        '''
        self.min_list=[]
        self.max_list=[]
        print('the shape of features is :',self.data.values.shape[1])
        for i in range(self.data.values.shape[1]):
            self.min_list.append(round(float(min(self.data[list(self.data.keys())[i]].values)),3))
            self.max_list.append(round(float(max(self.data[list(self.data.keys())[i]].values)),3))
        print('the max feature list is :',self.max_list)
        print('the min feature list is :', self.min_list)
        return self.max_list,self.min_list

    def normalization(self):
        for i,key in enumerate(list(self.data.keys())):
            if i: self.data[key]=round((self.data[key] - np.array(self.min[i])) / (np.array(self.max[i]) - np.array(self.min[i])), 6)

    def generator_(self):
        '''
        :return: yield the data of every time,
        shape:input_series:[time_size,field_size]
        label:[predict_size]
        '''

        if self.is_training:low,high=0,int(self.data.values.shape[0]*self.data_divide)
        else:low,high=int(self.data.values.shape[0]*self.data_divide),self.data.values.shape[0]

        while low <= high-(self.prediction_size+self.time_size):
            yield (np.array(self.data.values[low:low+self.time_size]),
                   self.data.values[low + self.time_size:low + self.time_size+self.prediction_size,-1])
            if self.is_training:low = low + self.window_step
            else:low=low+self.prediction_size
        return

    def next_batch(self,batch_size,epochs, is_training=True):
        '''
        :return the iterator!!!
        :param batch_size:
        :param epochs:
        :return:
        '''
        self.is_training=is_training
        dataset=tf.data.Dataset.from_generator(self.generator_,output_types=(tf.float32,tf.float32))

        if self.is_training:
            dataset=dataset.shuffle(buffer_size=(int(self.data.values.shape[0]*self.data_divide)-(self.time_size+self.prediction_size)//self.window_step+1))
            dataset=dataset.repeat(count=epochs)
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()

        return iterator.get_next()

if __name__=='__main__':
    iter=DataIterator(site_id=0)
    print(iter.data.keys())
    # print(iter.data.loc[iter.data['ZoneID']==0])
    next=iter.next_batch(32,1)
    for _ in range(4):
        x,y=tf.Session().run(next)
        print(x.shape)
        print(y.shape)