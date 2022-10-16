# -- coding: utf-8 --
# -- coding: utf-8 --
'''
the shape of sparsetensor is a tuuple, like this
(array([[  0, 297],
       [  0, 296],
       [  0, 295],
       ...,
       [161,   2],
       [161,   1],
       [161,   0]], dtype=int32), array([0.00323625, 0.00485437, 0.00323625, ..., 0.00646204, 0.00161551,
       0.00161551], dtype=float32), (162, 300))
axis=0: is nonzero values, x-axis represents Row, y-axis represents Column.
axis=1: corresponding the nonzero value.
axis=2: represents the sparse matrix shape.
'''

from __future__ import division
from __future__ import print_function
from models.utils import *
from models.models import GCN
from models.hyparameter import parameter
from models.embedding import embedding
from comparison_model_p.lstm import lstm
from models.data_next import DataClass
from comparison_model_p.trans.trans import BridgeTransformer
# from comparison_model_p.trans.lstm import lstm

import pandas as pd
import tensorflow as tf
import numpy as np
import os
import argparse

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"


class Model(object):
    def __init__(self, para):
        self.para = para
        self.adj = preprocess_adj(self.adjecent())

        # define gcn model
        if self.para.model_name == 'gcn_cheby':
            self.support = chebyshev_polynomials(self.adj, self.para.max_degree)
            self.num_supports = 1 + self.para.max_degree
            self.model_func = GCN
        else:
            self.support = [self.adj]
            self.num_supports = 1
            self.model_func = GCN

        # define placeholders
        self.placeholders = {
            'position': tf.placeholder(tf.int32, shape=(1, self.para.site_num), name='input_position'),
            'day': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_day'),
            'hour': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_hour'),
            'minute': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_minute'),
            'indices_i': tf.placeholder(dtype=tf.int64, shape=[None, None], name='input_indices'),
            'values_i': tf.placeholder(dtype=tf.float32, shape=[None], name='input_values'),
            'dense_shape_i': tf.placeholder(dtype=tf.int64, shape=[None], name='input_dense_shape'),
            'features_s': tf.placeholder(tf.float32, shape=[None, self.para.input_length, self.para.site_num, self.para.features], name='input_s'),
            'labels_s': tf.placeholder(tf.float32, shape=[None, self.para.site_num, self.para.output_length], name='labels_s'),
            'features_p': tf.placeholder(tf.float32, shape=[None, self.para.input_length, self.para.features_p], name='input_p'),
            'labels_p': tf.placeholder(tf.float32, shape=[None, self.para.output_length], name='labels_p'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='input_zero')  # helper variable for sparse dropout
        }
        self.supports = [tf.SparseTensor(indices=self.placeholders['indices_i'],
                                         values=self.placeholders['values_i'],
                                         dense_shape=self.placeholders['dense_shape_i']) for _ in range(self.num_supports)]
        self.model()

    def adjecent(self):
        '''
        :return: adj matrix
        '''
        data = pd.read_csv(filepath_or_buffer=self.para.file_adj)
        adj = np.zeros(shape=[self.para.site_num, self.para.site_num])
        for line in data[['src_FID', 'nbr_FID']].values:
            adj[line[0]][line[1]] = 1
        return adj

    def model(self):
        '''
        :param batch_size: 64
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: True
        :return:
        '''
        p_emd = embedding(self.placeholders['position'], vocab_size=self.para.site_num, num_units=self.para.emb_size,scale=False, scope="position_embed")
        p_emd = tf.reshape(p_emd, shape=[1, self.para.site_num, self.para.emb_size])
        self.p_emd = tf.tile(tf.expand_dims(p_emd, axis=0), [self.para.batch_size, self.para.input_length+self.para.output_length, 1, 1])

        d_emb = embedding(self.placeholders['day'], vocab_size=32, num_units=self.para.emb_size,scale=False, scope="day_embed")
        self.d_emd = tf.reshape(d_emb, shape=[self.para.batch_size, self.para.input_length + self.para.output_length,
                                              self.para.site_num, self.para.emb_size])

        h_emb = embedding(self.placeholders['hour'], vocab_size=24, num_units=self.para.emb_size,scale=False, scope="hour_embed")
        self.h_emd = tf.reshape(h_emb, shape=[self.para.batch_size, self.para.input_length + self.para.output_length,
                                              self.para.site_num, self.para.emb_size])

        m_emb = embedding(self.placeholders['minute'], vocab_size=4, num_units=self.para.emb_size,scale=False, scope="minute_embed")
        self.m_emd = tf.reshape(m_emb, shape=[self.para.batch_size, self.para.input_length + self.para.output_length,
                                              self.para.site_num, self.para.emb_size])

        if self.para.model_name=='lstm':
            # features=tf.layers.dense(self.placeholders['features'], units=self.para.emb_size) #[-1, site num, emb_size]
            features = tf.reshape(self.placeholders['features_p'], shape=[self.para.batch_size,
                                                                         self.para.input_length,
                                                                         self.para.features_p])

            # this step use to encoding the input series data
            '''
            lstm, return --- for example ,output shape is :(32, 3, 162, 128)
            axis=0: bath size
            axis=1: input data time size
            axis=2: numbers of the nodes
            axis=3: output feature size
            '''
            encoder_init = lstm(self.para.batch_size,
                                    predict_time=self.para.output_length,
                                    layer_num=self.para.hidden_layer,
                                    nodes=self.para.hidden_size,
                                    placeholders=self.placeholders)

            h_states= encoder_init.encoding(features)

            # decoder
            print('#................................in the decoder step......................................#')
            # this step to presict the polutant concentration
            self.pres=encoder_init.decoding(h_states)

        elif self.para.model_name=='trans':
            # features=tf.layers.dense(self.placeholders['features'], units=self.para.emb_size) #[-1, site num, emb_size]
            features = tf.reshape(self.placeholders['features_p'], shape=[self.para.batch_size,
                                                                         self.para.input_length,
                                                                         self.para.features_p])
            x = tf.layers.dense(inputs=features, units=self.para.emb_size, name='x_')
            # this step use to encoding the input series data
            '''
            lstm, return --- for example ,output shape is :(32, 3, 162, 128)
            axis=0: bath size
            axis=1: input data time size
            axis=2: numbers of the nodes
            axis=3: output feature size
            '''
            in_day = self.d_emd[:, :self.para.input_length, 0, :]
            in_hour = self.h_emd[:, :self.para.input_length, 0, :]
            in_minute = self.m_emd[:, :self.para.input_length, 0, :]
            in_time=tf.add_n([x, in_day, in_hour, in_minute])

            out_day = self.d_emd[:, self.para.input_length:, 0, :]
            out_hour = self.h_emd[:, self.para.input_length:, 0, :]
            out_minute = self.m_emd[:, self.para.input_length:, 0, :]
            out_time=tf.add_n([out_day, out_hour, out_minute])

            temporal = BridgeTransformer(arg=self.para)
            out_puts= temporal.encoder(hiddens=in_time, hidden=out_time)

            # encoder_init = lstm(self.para.batch_size,
            #                     predict_time=self.para.output_length,
            #                     layer_num=self.para.hidden_layer,
            #                     nodes=self.para.hidden_size,
            #                     placeholders=self.placeholders)
            # h_states = encoder_init.encoding(x)
            # h_states = encoder_init.decoding(h_states,time=out_x)

            # out_puts = tf.concat([h_states, out_puts],axis=-1)
            results_pollution=tf.layers.dense(inputs=out_puts, units=64, name='layer_pollution_1')
            results_pollution=tf.layers.dense(inputs=results_pollution, units=1, name='layer_pollution_2')
            results_pollution=tf.squeeze(results_pollution,axis=-1)
            self.pres=results_pollution

        print('pres shape is : ', self.pres.shape)

        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(self.pres + 1e-10 - self.placeholders['labels_p']), axis=0)))
        # weights=tf.Variable(initial_value=tf.constant(value=0.5, dtype=tf.float32),name='loss_weight')
        # self.cross_entropy = loss1 + loss2
        # backprocess and update the parameters
        # self.train_op = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.cross_entropy)
        self.train_op = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.loss)

        print('#...............................in the training step.....................................#')

    def test(self):
        '''
        :param batch_size: usually use 1
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: False
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def re_current(self, a, max, min):
        return [num * (max - min) + min for num in a]

    def run_epoch(self):
        '''
        from now on,the model begin to training, until the epoch to 100
        '''

        max_mae = 100
        self.sess.run(tf.global_variables_initializer())
        iterate = DataClass(self.para)

        train_next = iterate.next_batch(batch_size=self.para.batch_size, epoch=self.para.epoch, is_training=True)

        for i in range(int((iterate.length // self.para.site_num * self.para.divide_ratio - (
                self.para.input_length + self.para.output_length)) // self.para.step)
                       * self.para.epoch // self.para.batch_size):
            x_s, day, hour, minute, label_s, x_p, label_p = self.sess.run(train_next)
            x_s = np.reshape(x_s, [-1, self.para.input_length, self.para.site_num, self.para.features])
            day = np.reshape(day, [-1, self.para.site_num])
            hour = np.reshape(hour, [-1, self.para.site_num])
            minute = np.reshape(minute, [-1, self.para.site_num])
            feed_dict = construct_feed_dict(x_s, self.adj, label_s, day, hour, minute, x_p, label_p, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.para.dropout})

            loss, _ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss))

            # validate processing
            if i % 100 == 0:
                mae = self.evaluate()
                if max_mae > mae:
                    print("the validate average loss value is : %.6f" % (mae))
                    max_mae = mae
                    self.saver.save(self.sess, save_path=self.para.save_path + 'model.ckpt')

    def evaluate(self):
        '''
        :param para:
        :param pre_model:
        :return:
        '''
        label_s_list, label_p_list = list(), list()
        pre_s_list, pre_p_list = list(),list()

        # with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(self.para.save_path)
        if not self.para.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)
            # self.saver.save(self.sess, save_path='gcn/model/' + 'model.ckpt')

        iterate_test = DataClass(hp=self.para)
        test_next = iterate_test.next_batch(batch_size=self.para.batch_size, epoch=1, is_training=False)

        max_s, min_s = iterate_test.max_s['speed'], iterate_test.min_s['speed']
        max_p, min_p = iterate_test.max_p['AQI'], iterate_test.min_p['AQI']

        # '''
        for i in range(int((iterate_test.length // self.para.site_num
                            - iterate_test.length // self.para.site_num * iterate_test.divide_ratio
                            - (self.para.input_length + self.para.output_length)) // iterate_test.output_length) // self.para.batch_size):
            x_s, day, hour, minute, label_s, x_p, label_p = self.sess.run(test_next)
            x_s = np.reshape(x_s, [-1, self.para.input_length, self.para.site_num, self.para.features])
            day = np.reshape(day, [-1, self.para.site_num])
            hour = np.reshape(hour, [-1, self.para.site_num])
            minute = np.reshape(minute, [-1, self.para.site_num])
            feed_dict = construct_feed_dict(x_s, self.adj, label_s, day, hour, minute, x_p, label_p, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: 0.0})

            pre_p = self.sess.run((self.pres), feed_dict=feed_dict)
            label_p_list.append(label_p)
            pre_p_list.append(pre_p)

        label_p_list = np.reshape(np.array(label_p_list, dtype=np.float32), [-1])
        pre_p_list = np.reshape(np.array(pre_p_list, dtype=np.float32), [-1])
        if self.para.normalize:
            label_p_list = np.array(self.re_current(label_p_list, max_p, min_p))
            pre_p_list = np.array(self.re_current(pre_p_list, max_p, min_p))

        label_p_list = np.reshape(label_p_list, [-1])
        pre_p_list = np.reshape(pre_p_list, [-1])
        mae, rmse, mape, cor, r2 = metric(label_p_list, pre_p_list)  # 产生预测指标
        # describe(label_list, predict_list)   #预测值可视化
        return mae


def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    print('#......................................beginning........................................#')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')

    if int(val) == 1:
        para.is_training = True
    else:
        para.batch_size = 1
        para.is_training = False

    pre_model = Model(para)
    pre_model.initialize_session()

    if int(val) == 1:
        pre_model.run_epoch()
    else:
        pre_model.evaluate()

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()