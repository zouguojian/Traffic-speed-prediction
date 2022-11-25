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
from models.encoder import Encoder_ST
from models.decoder import Decoder_ST
from models.bridge import BridgeTransformer
from models.bridge_lstm import LstmClass
from models.inference import InferenceClass
from models.data_next import DataClass
from models.bridge import transformAttention

import pandas as pd
import tensorflow as tf
import numpy as np
import os
import argparse
import datetime

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"
#
# os.environ['CUDA_VISIBLE_DEVICES']='3'
#
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


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

        # encoder
        print('#................................in the encoder step....................................#')
        with tf.variable_scope(name_or_scope='encoder'):
            '''
            return, the gcn output --- for example, inputs.shape is :  (32, 3, 162, 32)
            axis=0: bath size
            axis=1: input data time size
            axis=2: numbers of the nodes
            axis=3: output feature size
            '''
            timestamp = [self.h_emd, self.m_emd]
            position = self.p_emd
            # [-1, input_length, site num, emb_size]

            if self.para.model_name == 'STGIN_1':
                speed = FC(self.placeholders['features_s'], units=[self.para.emb_size, self.para.emb_size], activations=[tf.nn.relu, None],
                            bn=False, bn_decay=0.99, is_training=self.para.is_training)
            else:
                speed = tf.transpose(self.placeholders['features_s'],perm=[0, 2, 1, 3])
                speed = tf.reshape(speed, [-1, self.para.input_length, self.para.features])
                speed3 = tf.layers.conv1d(inputs=speed,
                                         filters=self.para.emb_size,
                                         kernel_size=3,
                                         padding='SAME',
                                         kernel_initializer=tf.truncated_normal_initializer(),
                                         name='conv_1')

                speed2 = tf.layers.conv1d(inputs=tf.reverse(speed,axis=[1]),
                                         filters=self.para.emb_size,
                                         kernel_size=3,
                                         padding='SAME',
                                         kernel_initializer=tf.truncated_normal_initializer(),
                                         name='conv_2')

                speed1 = tf.layers.conv1d(inputs=speed,
                                         filters=self.para.emb_size,
                                         kernel_size=1,
                                         padding='SAME',
                                         kernel_initializer=tf.truncated_normal_initializer(),
                                         name='conv_3')
                # speed2 = tf.nn.sigmoid(speed2)
                speed2 = tf.reverse(speed2,axis=[1])
                speed2 = tf.multiply(speed2, tf.nn.sigmoid(speed2))
                speed3 = tf.multiply(speed3, tf.nn.sigmoid(speed3))
                speed = tf.add_n([speed1, speed2, speed3])
                speed = tf.reshape(speed, [-1, self.para.site_num, self.para.input_length, self.para.emb_size])
                speed = tf.transpose(speed, perm=[0, 2, 1, 3])
            # [-1, input_length, emb_size]
            STE = STEmbedding(position, timestamp, 0, self.para.emb_size, False, 0.99, self.para.is_training)

            encoder = Encoder_ST(hp=self.para, placeholders=self.placeholders, model_func=self.model_func)
            encoder_outs = encoder.encoder_spatio_temporal(speed = speed,
                                                           STE = STE[:, :self.para.input_length,:,:],
                                                           supports=self.supports)
            print('encoder encoder_outs shape is : ', encoder_outs.shape)

        # inference
        print('#................................in the inference step...................................#')
        with tf.variable_scope(name_or_scope='inference'):
            inference=InferenceClass(para=self.para)
            self.pres_s= inference.dynamic_inference(features=encoder_outs, STE=STE[:, self.para.input_length:,:,:])
            print('pres_s shape is : ', self.pres_s.shape)

        self.loss1 = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(self.pres_s + 1e-10 - self.placeholders['labels_s']), axis=0)))
        self.train_op_1 = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.loss1)

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

            loss_1, _ = self.sess.run((self.loss1, self.train_op_1), feed_dict=feed_dict)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss_1))

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
        label_s_list, pre_s_list = list(), list()

        # with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(self.para.save_path)
        if not self.para.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)
            # self.saver.save(self.sess, save_path='gcn/model/' + 'model.ckpt')

        iterate_test = DataClass(hp=self.para)
        test_next = iterate_test.next_batch(batch_size=self.para.batch_size, epoch=1, is_training=False)
        max_s, min_s = iterate_test.max_s['speed'], iterate_test.min_s['speed']

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
            if i == 0: begin_time = datetime.datetime.now()
            pre_s= self.sess.run((self.pres_s), feed_dict=feed_dict)
            if i == 0:
                end_t = datetime.datetime.now()
                total_t = end_t - begin_time
                print("Total running times is : %f" % total_t.total_seconds())
            label_s_list.append(label_s)
            pre_s_list.append(pre_s)

        label_s_list = np.reshape(np.array(label_s_list, dtype=np.float32),
                                [-1, self.para.site_num, self.para.output_length]).transpose([1, 0, 2])
        pre_s_list = np.reshape(np.array(pre_s_list, dtype=np.float32),
                                  [-1, self.para.site_num, self.para.output_length]).transpose([1, 0, 2])
        if self.para.normalize:
            label_s_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max_s, min_s) for site_label in label_s_list])
            pre_s_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max_s, min_s) for site_label in pre_s_list])
        else:
            label_s_list = np.array([np.reshape(site_label, [-1]) for site_label in label_s_list])
            pre_s_list = np.array([np.reshape(site_label, [-1]) for site_label in pre_s_list])
        print('speed prediction result')
        label_all = np.reshape(np.array(label_s_list),newshape=[self.para.site_num, -1, self.para.output_length])
        predict_all = np.reshape(np.array(pre_s_list), newshape=[self.para.site_num, -1, self.para.output_length])

        label_s_list = np.reshape(label_s_list, [-1])
        pre_s_list = np.reshape(pre_s_list, [-1])
        mae, rmse, mape, cor, r2 = metric(pre_s_list, label_s_list)  # 产生预测指标

        for i in range(self.para.output_length):
            print('in the %d time step, the evaluating indicator'%(i+1))
            metric(np.reshape(predict_all[:,:,i], [-1]), np.reshape(label_all[:,:,i], [-1]))

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