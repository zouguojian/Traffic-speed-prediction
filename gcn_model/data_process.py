# -- coding: utf-8 --

import tensorflow as tf
import numpy as np
from gcn_model.data_read import *
import argparse
from gcn_model.hyparameter import parameter
import os

train_path=r'data/training_data/train.csv'
train_p_path=r'data/pollution/train_p.csv'

class DataIterator():             #切记这里的训练时段和测试时段的所用的对象不变，否则需要重复加载数据
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
        self.min_value=0.000000000001
        self.site_id=site_id                   # ozone ID
        self.time_size=time_size               # time series length of input
        self.prediction_size=prediction_size   # the length of prediction
        self.is_training=is_training           # true or false
        self.data_divide=data_divide           # the divide between in training set and test set ratio
        self.window_step=window_step           # windows step
        self.para=hp
        self.source_data=self.get_source_data(train_path)
        self.source_train_p=self.get_source_data(train_p_path)

        # self.data=self.source_data.loc[self.source_data['ZoneID']==self.site_id]
        self.id_dict = dict()
        self.data=self.source_data
        self.data_p=self.source_train_p
        self.id_index=dict()

        # 路字典，用以记录收费站之间是否有路存在
        for line in self.data.values:
            if (line[0], line[1]) not in self.id_dict and line[0] != line[1]:
                self.id_dict[(int(line[0]), int(line[1]))] = 1

        # 生成邻接矩阵
        self.adj = np.zeros(shape=[len(self.id_dict), len(self.id_dict)],dtype=np.int32)
        self.adjacent(self.adj)

        self.length=self.data.values.shape[0]  # data length
        self.max_list,self.min_list=self.get_max_min(self.data)   # max and min are list type, used for the later normalization
        self.max_list_p,self.min_list_p=self.get_max_min(self.data_p) # pollution

        self.normalize=normalize
        if self.normalize:
            self.normalization(data=self.data,index=6,max_list=self.max_list,min_list=self.min_list) # normalization
            self.normalization(data=self.data_p,index=1,max_list=self.max_list_p,min_list=self.min_list_p)

    def adjacent(self,adj):
        node_ids=dict()
        index=0

        # 定义节点，即节点id化，路即为节点
        for id in sorted(self.id_dict):
            node_ids[(id[0], id[1])]=index
            index+=1

        # 邻接矩阵初始化
        for i,node_i in enumerate(node_ids):
            for j,node_j in enumerate(node_ids):
                if node_i==node_j:continue
                if node_j[1] == node_i[0]: adj[i][j] = 1
                # if node_i[0]==node_j[0] or node_i[1]==node_j[0]: adj[i][j]=1

        if not os.path.exists('data/training_data/adjacent.csv'):
            file = open('data/training_data/adjacent.csv', 'w', encoding='utf-8')
            writer = csv.writer(file)
            writer.writerow(['src_FID', 'nbr_FID'])
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i][j]==1:
                        writer.writerow([i,j])
            file.close()

    def get_source_data(self,file_path=None):
        '''
        :return:
        '''
        data = pd.read_csv(file_path, encoding='utf-8')
        return data

    def get_max_min(self,data=None):
        '''
        :return: the max and min value of input features
        '''
        min_list=[]
        max_list=[]
        # print('the shape of features is :',self.data.values.shape[1])
        for i in range(data.values.shape[1]):
            min_list.append(min(data[list(data.keys())[i]].values))
            max_list.append(max(data[list(data.keys())[i]].values))
        print('the max feature list is :',max_list)
        print('the min feature list is :', min_list)
        return max_list,min_list

    def normalization(self,data=None,index=1,max_list=[],min_list=[]):
        keys=list(data.keys())

        for i in range(index,len(keys)):
            data[keys[i]]=(data[keys[i]] - np.array(min_list[i])) / (np.array(max_list[i]) - np.array(min_list[i]+self.min_value))

    def generator_(self):
        '''
        :return: yield the data of every time,
        shape:input_series:[time_size,field_size]
        label:[predict_size]
        '''
        para=self.para

        if self.is_training:
            low,high=24*6*para.site_num, int(self.data.shape[0]//para.site_num * self.data_divide)*para.site_num
            low_p=24*6
        else:
            low,high=int(self.data.shape[0]//para.site_num * self.data_divide) *para.site_num, self.data.shape[0]
            low_p=int(self.data_p.shape[0] * self.data_divide)

        while low+para.site_num*(para.input_length + para.output_length)<= high:
            label=self.data.values[low + self.time_size * para.site_num: low + (self.time_size + self.prediction_size) * para.site_num,-2:-1]
            label=np.concatenate([label[i * para.site_num:(i + 1) * para.site_num, :] for i in range(self.prediction_size)], axis=1)

            yield (self.data.values[low:low+self.time_size*para.site_num, 6:7],
                   self.data.values[low:low+(self.time_size+self.prediction_size)*para.site_num, 4],
                   self.data.values[low:low + (self.time_size+self.prediction_size)* para.site_num, 5],
                   label,
                   self.data_p.values[low_p:low_p+self.time_size,1:],
                   self.data_p.values[low_p+self.time_size: low_p+self.time_size+self.prediction_size,1])
            if self.is_training:
                low += self.window_step*para.site_num
                low_p+=self.window_step
            else:
                low+=self.prediction_size*para.site_num
                low_p+=self.prediction_size
        return

    def next_batch(self,batch_size,epochs, is_training=True):
        '''
        :return the iterator!!!
        :param batch_size:
        :param epochs:
        :return:
        '''
        self.is_training=is_training
        dataset=tf.data.Dataset.from_generator(self.generator_,output_types=(tf.float32,tf.int32, tf.int32, tf.float32, tf.float32, tf.float32))

        if self.is_training:
            dataset=dataset.shuffle(buffer_size=int(self.data.values.shape[0]//self.para.site_num * self.data_divide-self.time_size-self.prediction_size)//self.window_step)
            dataset=dataset.repeat(count=epochs)
        dataset=dataset.batch(batch_size=batch_size)
        iterator=dataset.make_one_shot_iterator()

        return iterator.get_next()
# #
if __name__=='__main__':
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    iter=DataIterator(site_id=0,normalize=False,hp=para, time_size=6, prediction_size=3)
    print(iter.data.keys())
    # print(iter.data.loc[iter.data['ZoneID']==0])
    next=iter.next_batch(1,1, False)
    with tf.Session() as sess:
        for _ in range(4):
            x,y=sess.run(next)
            print(x.shape)
            print(y.shape)
            print(x[0])
            print(y[0])