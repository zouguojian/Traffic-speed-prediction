# -- coding: utf-8 --
import tensorflow as tf
'''
noted that , for PSPNN, we do not use weather or other additional data,
we just use the PSPNN model to extract the spatio-temporal correlation of input traffic data,
and to predict long-term traffic speed.
if you have weather or other data, you can added it to model, thanks.
'''
class cnn_bilstm(object):
    def __init__(self, batch_size, predict_time,layer_num=1, nodes=64, placeholders=None):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.placeholders=placeholders
        self.predict_time=predict_time
        self.h=3
        self.w=3
        self.position_size=108
        self.decoder()

    def cnn(self,x):
        '''
        :param x: shape is [batch size * input length, site num, features]
        :return: shape is [batch size, height, channel]
        '''

        # x=tf.expand_dims(x, axis=-1) # [batch size * site num, input length, features, 1]

        filter1=tf.get_variable("filter1", [self.h,self.w,1,32], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer1=tf.nn.conv2d(input=x,filter=filter1,strides=[1,1,1,1],padding='SAME')
        # bn1=tf.layers.batch_normalization(layer1,training=self.placeholders['is_training'])
        relu1=tf.nn.relu(layer1)
        # max_pool1=tf.nn.max_pool(relu1, ksize=[self.h,self.w, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # print('max_pool1 output shape is : ',max_pool1.shape)

        filter2 = tf.get_variable("filter2", [self.h,self.w,32,64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer2 = tf.nn.conv2d(input=relu1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME')
        # bn2=tf.layers.batch_normalization(layer2,training=self.placeholders['is_training'])
        relu2=tf.nn.relu(layer2)
        # max_pool2=tf.nn.max_pool(relu2, ksize=[self.h,self.w, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # print('max_pool2 output shape is : ',max_pool2.shape)

        filter3 = tf.get_variable("filter3", [self.h,self.w,64,64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer3 = tf.nn.conv2d(input=relu2, filter=filter3, strides=[1, 1, 1, 1], padding='SAME')
        # bn3=tf.layers.batch_normalization(layer3,training=self.placeholders['is_training'])
        relu3=tf.nn.relu(layer3)
        # max_pool3=tf.nn.max_pool(relu3, ksize=[self.h,self.w, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # print('max_pool3 output shape is : ', max_pool3.shape)

        cnn_shape = relu3.get_shape().as_list()
        nodes = cnn_shape[2] * cnn_shape[3]
        # reshaped = tf.reshape(max_pool3, [cnn_shape[0], nodes])
        '''shape is  : [batch size, site num, features, channel]'''
        s=tf.reshape(relu3, shape=[cnn_shape[0],cnn_shape[1],nodes])
        # res=tf.reduce_mean(res,axis=3)
        print('s shape is : ',s.shape)

        # print('cnn output shape is : ',s.shape)
        return s

    def decoder(self):
        '''
        :return:  shape is [batch size, time size, hidden size]
        '''

        def cell():
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)  # single lstm unit
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=1-self.placeholders['dropout'])

            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)  # single lstm unit
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=1-self.placeholders['dropout'])

            return cell_fw, cell_bw

        cell_fw, cell_bw=cell()

        self.df_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell_fw for _ in range(self.layer_num)])
        self.db_mlstm = tf.nn.rnn_cell.MultiRNNCell([cell_bw for _ in range(self.layer_num)])

    def decoding(self,  x):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''

        x_shape = x.get_shape().as_list()
        x=self.cnn(x=tf.reshape(x, shape=[x_shape[0], x_shape[1], x_shape[2],x_shape[3]]))
        x=tf.reshape(x, shape=[x_shape[0], x_shape[1], x_shape[2], self.nodes])

        x=tf.transpose(x,perm=[0, 2, 1, 3])
        x=tf.reshape(x,shape=[-1, x_shape[1], self.nodes]) # [batch * site num, input leangth, nodes]
        print(x.shape)

        with tf.variable_scope('encoder_lstm'):

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.df_mlstm, self.db_mlstm, x,
                                                                        dtype=tf.float32)  # [2, batch_size, seq_length, output_size]
            outputs = tf.concat(outputs, axis=2)
            print(outputs.shape)

            results_speed = tf.layers.dense(inputs=outputs[:,-1,:], units=64, activation=tf.nn.relu, name='layer_spped_1', reuse=tf.AUTO_REUSE)
            results = tf.layers.dense(inputs=results_speed, units=self.predict_time, activation=tf.nn.relu, name='layer_speed_2', reuse=tf.AUTO_REUSE)
            # results = tf.layers.dense(inputs=outputs[:,-1,:], units=self.predict_time, name='layer', reuse=tf.AUTO_REUSE)

        return tf.reshape(results, shape=[-1, x_shape[2], self.predict_time])

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