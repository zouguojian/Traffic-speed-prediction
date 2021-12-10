# -- coding: utf-8 --

import tensorflow as tf

class st_cnn(object):
    def __init__(self, batch_size, predict_time, placeholders=None):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size=batch_size
        self.placeholders=placeholders
        self.predict_time=predict_time
        self.h=3
        self.w=3


    def cnn(self,x):
        '''
        :param x: shape is [batch size,  input length, features]
        :return: shape is [batch size, height, channel]
        '''
        x=tf.expand_dims(x,axis=-1) # [batch size, input length, features, channel]

        filter1=tf.get_variable("filter1", [self.h,self.w,1,64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer1=tf.nn.conv2d(input=x,filter=filter1,strides=[1,1,1,1],padding='SAME')
        # bn1=tf.layers.batch_normalization(layer1,training=self.placeholders['is_training'])
        relu1=tf.nn.relu(layer1)
        # max_pool1=tf.nn.max_pool(relu1, ksize=[self.h,self.w, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # print('max_pool1 output shape is : ',max_pool1.shape)

        filter2 = tf.get_variable("filter2", [self.h,self.w,64,64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer2 = tf.nn.conv2d(input=relu1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME')
        # bn2=tf.layers.batch_normalization(layer2,training=self.placeholders['is_training'])
        relu2=tf.nn.relu(layer2)
        # max_pool2=tf.nn.max_pool(relu2, ksize=[self.h,self.w, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # print('max_pool2 output shape is : ',max_pool2.shape)

        filter3 = tf.get_variable("filter3", [self.h,self.w,64,128], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer3 = tf.nn.conv2d(input=relu2, filter=filter3, strides=[1, 1, 1, 1], padding='SAME')
        # bn3=tf.layers.batch_normalization(layer3,training=self.placeholders['is_training'])
        relu3=tf.nn.relu(layer3)
        # max_pool3=tf.nn.max_pool(relu3, ksize=[self.h,self.w, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # filter_=tf.get_variable("filter", [1,1,1,128], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # res = tf.nn.conv2d(input=x, filter=filter_, strides=[1, 1, 1, 1], padding='SAME')

        res=tf.add_n([relu3])

        # print('max_pool3 output shape is : ', max_pool3.shape)

        cnn_shape = res.get_shape().as_list()
        nodes = cnn_shape[1]*cnn_shape[2] * cnn_shape[3]
        # reshaped = tf.reshape(max_pool3, [cnn_shape[0], nodes])
        '''shape is  : [batch size, site num, features, channel]'''
        res=tf.reshape(res, shape=[cnn_shape[0],nodes])
        # res=tf.reduce_mean(res,axis=3)
        print('relu3 shape is : ',res.shape)

        s=tf.layers.dense(inputs=res, units=self.predict_time)

        # print('cnn output shape is : ',s.shape)
        return s

    def decoding(self, x):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''

        pre=self.cnn(x=x)

        return pre

import numpy as np
if __name__ == '__main__':
    train_data=np.random.random(size=[32,1,162,5])
    x=tf.placeholder(tf.float32, shape=[32, 3, 162,5])
    r=cnn_lstm(32,10,2,128)
    # hs=r.cnn(x)

    model_cnn_ = []
    for time in range(3):
        model_cnn_.append(r.cnn(tf.expand_dims(x[:,time,:,:],axis=1)))
    inputs = tf.concat(values=model_cnn_, axis=1)
    print(inputs.shape)

    pos_es=r.position_em()
    pos_es=tf.expand_dims(pos_es,axis=0)
    pos_es=tf.tile(pos_es,multiples=[32,3,1,1])
    pos_es = tf.layers.dense(inputs=pos_es, units=128, name='layer', activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
    print(pos_es.shape)

    # print(hs.shape)
    #
    # pre=r.decoding(hs)
    # print(pre.shape)