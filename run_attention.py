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
from gcn_model.utils import *
from gcn_model.models import GCN
from gcn_model.hyparameter import parameter
from gcn_model.gat import embedding
from gcn_model.gat import Transformer

import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
import model.decoder as decoder
import matplotlib.pyplot as plt
import model.normalization as normalization
import model.encoder as encoder
import model.encoder_lstm as encoder_lstm
import gcn_model.data_process as data_load
import os
import argparse
import shutil

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
            'indices_i': tf.placeholder(dtype=tf.int64, shape=[None, None],name='input_indices'),
            'values_i': tf.placeholder(dtype=tf.float32, shape=[None],name='input_values'),
            'dense_shape_i': tf.placeholder(dtype=tf.int64, shape=[None],name='input_dense_shape'),
            # None : batch _size * time _size
            'features': tf.placeholder(tf.float32, shape=[None, self.para.site_num, self.para.features],
                                       name='input_features'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.para.site_num, self.para.output_length],
                                     name='labels'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='input_zero')  # helper variable for sparse dropout
        }

        self.supports= [tf.SparseTensor(indices=self.placeholders['indices_i'],
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

        '''
        feedforward and BN layer
        output shape:[batch, time_size,field_size,new_features]
        '''
        # normal=normalization.Normalization(inputs=self.placeholders['features'],out_size=self.para.features,is_training=self.para.is_training)
        # normal.normal()

        # with tf.variable_scope('position_gcn'):
        #     self.p_emb = embedding(self.placeholders['position'], vocab_size=self.para.site_num, num_units=self.para.position_units,
        #                             scale=True, scope="position_embed")
        #     p_gcn=self.model_func(self.placeholders, input_dim=self.para.position_units, para=self.para)
        #     p_emd=p_gcn.predict(self.p_emb)
        #     p_emd = tf.reshape(p_emd, shape=[1, self.para.site_num, self.para.gcn_output_size])
        #     p_emd=tf.expand_dims(p_emd,axis=0)
        #     p_emd=tf.tile(p_emd,[self.para.batch_size, self.para.input_length, 1, 1])
        #     print('p_emd shape is : ', p_emd.shape)

        with tf.variable_scope('position'):
            p_emd = embedding(self.placeholders['position'], vocab_size=self.para.site_num,
                              num_units=self.para.position_units,
                              scale=False, scope="position_embed")
            p_emd = tf.reshape(p_emd, shape=[1, self.para.site_num, self.para.position_units])
            p_emd = tf.expand_dims(p_emd, axis=0)
            self.p_emd = tf.tile(p_emd, [self.para.batch_size, self.para.input_length, 1, 1])
            print('p_emd shape is : ', self.p_emd.shape)

        with tf.variable_scope('day'):
            self.d_emb = embedding(self.placeholders['day'], vocab_size=32, num_units=self.para.gcn_output_size,
                                   scale=False, scope="day_embed")
            self.d_emd = tf.reshape(self.d_emb,
                                    shape=[self.para.batch_size, self.para.input_length + self.para.output_length,
                                           self.para.site_num, self.para.gcn_output_size])
            print('d_emd shape is : ', self.d_emd.shape)

        with tf.variable_scope('hour'):
            self.h_emb = embedding(self.placeholders['hour'], vocab_size=24, num_units=self.para.gcn_output_size,
                                   scale=False, scope="hour_embed")
            self.h_emd = tf.reshape(self.h_emb,
                                    shape=[self.para.batch_size, self.para.input_length + self.para.output_length,
                                           self.para.site_num, self.para.gcn_output_size])
            print('h_emd shape is : ', self.h_emd.shape)

        # create model

        if self.para.model_name == 'gcn':
            '''
            return, the gcn output --- for example, inputs.shape is :  (batch, 3, 162, 32)
            axis=0: bath size
            axis=1: input data time size
            axis=2: numbers of the nodes
            axis=3: output feature size
            '''
            with tf.variable_scope('speed_gcn'):
                s_gcn = self.model_func(self.placeholders, input_dim=self.para.features, para=self.para)
                s_emd = s_gcn.predict(self.placeholders['features'])
                s_emd = tf.reshape(s_emd, shape=[self.para.batch_size, self.para.input_length, self.para.site_num,
                                                 self.para.gcn_output_size])
                print('s_emd shape is : ', s_emd.shape)

            # s_p_emd=tf.add_n([s_emd, self.p_emd, self.d_emd, self.h_emd])
            # transformer
            m = Transformer(self.para)
            x = m.encoder(speed=self.placeholders['features'], day=self.d_emd, hour=self.h_emd, position=self.p_emd)

            # this step use to encoding the input series data
            '''
            rlstm, return --- for example ,output shape is :(32, 3, 162, 128)
            axis=0: bath size
            axis=1: input data time size
            axis=2: numbers of the nodes
            axis=3: output feature size
            '''
            encoder_init = encoder_lstm.lstm(self.para.batch_size * self.para.site_num,
                                             self.para.hidden_layer,
                                             self.para.hidden_size,
                                             self.para.is_training,
                                             placeholders=self.placeholders)
            inputs = tf.reshape(x, shape=[self.para.batch_size, self.para.input_length, self.para.site_num,
                                          self.para.gcn_output_size])
            inputs = inputs + s_emd + self.p_emd
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
            inputs = tf.reshape(inputs, shape=[self.para.batch_size * self.para.site_num, self.para.input_length,
                                               self.para.gcn_output_size])

            h_states, c_states = encoder_init.encoding(inputs)
            h_states = tf.reshape(h_states, shape=[self.para.batch_size, self.para.site_num, self.para.input_length,
                                                   self.para.hidden_size])
            h_states = tf.transpose(h_states, perm=[0, 2, 1, 3])  # [batch, input length, site num, hidden size]

            print('h_states shape is : ', h_states.shape)

            # this step to presict the polutant concentration
            '''
            decoder, return --- for example ,output shape is :(32, 162, 1)
            axis=0: bath size
            axis=1: numbers of the nodes
            axis=2: label size
            '''
            decoder_init = decoder.lstm(self.para.batch_size * self.para.site_num,
                                        self.para.output_length,
                                        self.para.hidden_layer,
                                        self.para.hidden_size,
                                        placeholders=self.placeholders)
            inputs = tf.transpose(h_states, perm=[0, 2, 1, 3])
            inputs = tf.reshape(inputs, shape=[self.para.batch_size * self.para.site_num, self.para.input_length,
                                               self.para.hidden_size])
            pres = decoder_init.decoding(inputs)
            self.pres = tf.reshape(pres, shape=[self.para.batch_size, self.para.site_num, self.para.output_length])
            print('pres shape is : ', self.pres.shape)

            self.cross_entropy = tf.reduce_mean(
                tf.sqrt(tf.reduce_mean(tf.square(self.pres + 1e-10 - self.placeholders['labels']), axis=0)))

        elif self.para.model_name == 'gcn_encoder_decoder':
            with tf.variable_scope('position_gcn'):
                p_emb = tf.reshape(self.p_emd, shape=[-1, self.para.site_num, self.para.position_units])
                p_gcn = self.model_func(self.placeholders,
                                        input_dim=self.para.position_units,
                                        para=self.para,
                                        day=self.d_emd[:, :self.para.input_length, :, :],
                                        hour=self.h_emd[:, :self.para.input_length, :, :],
                                        position=self.p_emd,
                                        supports=self.supports)
                p_emd = p_gcn.predict(p_emb)
                self.g_p_emd = tf.reshape(p_emd, shape=[self.para.batch_size,
                                                        self.para.input_length,
                                                        self.para.site_num,
                                                        self.para.gcn_output_size])
                print('p_emd shape is : ', self.g_p_emd.shape)
            # encoder
            print('#................................in the encoder step......................................#')
            with tf.variable_scope(name_or_scope='encoder'):
                '''
                return, the gcn output --- for example, inputs.shape is :  (32, 3, 162, 32)
                axis=0: bath size
                axis=1: input data time size
                axis=2: numbers of the nodes
                axis=3: output feature size
                '''
                in_day = self.d_emd[:, :self.para.input_length, :, :]
                in_hour = self.h_emd[:, :self.para.input_length, :, :]
                encoder_gcn = self.model_func(self.placeholders,
                                              input_dim=self.para.features,
                                              para=self.para,
                                              day=in_day,
                                              hour=in_hour,
                                              position=self.p_emd,
                                              supports=self.supports)
                encoder_outs = encoder_gcn.predict(self.placeholders['features'])
                encoder_outs = tf.reshape(encoder_outs, shape=[self.para.batch_size,
                                                               self.para.input_length,
                                                               self.para.site_num,
                                                               self.para.gcn_output_size])
                print('encoder gcn outs shape is : ', encoder_outs.shape)

                # transformer
                m = Transformer(self.para)
                x = m.encoder(speed=self.placeholders['features'], day=in_day, hour=in_hour, position=self.p_emd)

                # this step use to encoding the input series data
                '''
                rlstm, return --- for example ,output shape is :(32, 3, 162, 128)
                axis=0: bath size
                axis=1: input data time size
                axis=2: numbers of the nodes
                axis=3: output feature size
                '''
                encoder_init = encoder_lstm.lstm(self.para.batch_size * self.para.site_num,
                                                 self.para.hidden_layer,
                                                 self.para.hidden_size,
                                                 self.para.is_training,
                                                 placeholders=self.placeholders)

                inputs = tf.reshape(x, shape=[self.para.batch_size, self.para.input_length, self.para.site_num,
                                              self.para.gcn_output_size])
                # trick
                inputs = inputs + encoder_outs + self.p_emd
                inputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
                inputs = tf.reshape(inputs, shape=[self.para.batch_size * self.para.site_num, self.para.input_length,
                                                   self.para.gcn_output_size])

                h_states, c_states = encoder_init.encoding(inputs)
                h_states = tf.reshape(h_states, shape=[self.para.batch_size, self.para.site_num, self.para.input_length,
                                                       self.para.hidden_size])
                h_states = tf.transpose(h_states, perm=[0, 2, 1, 3])

                print('encoder h states shape is : ', h_states.shape)

            # decoder
            print('#................................in the decoder step......................................#')
            with tf.variable_scope(name_or_scope='decoder'):
                '''
                return, the gcn output --- for example, inputs.shape is :  (32, 1, 162, 32)
                axis=0: bath size
                axis=1: input data time size
                axis=2: numbers of the nodes
                axis=3: output feature size
                '''
                out_day = self.d_emd[:, self.para.input_length:, :, :]
                out_hour = self.h_emd[:, self.para.input_length:, :, :]

                decoder_gcn = self.model_func(self.placeholders,
                                              input_dim=self.para.hidden_size,
                                              para=self.para,
                                              day=out_day,
                                              hour=out_hour,
                                              position=self.p_emd,
                                              supports=self.supports)

                # this step to presict the polutant concentration
                '''
                decoder, return --- for example ,output shape is :(32, 162, 1)
                axis=0: bath size
                axis=1: numbers of the nodes
                axis=2: label size
                '''
                decoder_init = decoder.lstm(self.para.batch_size * self.para.site_num,
                                            self.para.output_length,
                                            self.para.hidden_layer,
                                            self.para.hidden_size,
                                            placeholders=self.placeholders)

            self.pres = decoder_init.gcn_decoding(h_states, gcn=decoder_gcn, site_num=self.para.site_num)
            print('the pre output shape is : ',self.pres.shape)
            self.cross_entropy = tf.reduce_mean(
                tf.sqrt(tf.reduce_mean(tf.square(self.pres + 1e-10 - self.placeholders['labels']), axis=0)))

        elif self.para.model_name == 'lstm':
            # this step use to encoding the input series data
            '''
            rlstm, return --- for example ,output shape is :(32, 3, 162, 128)
            axis=0: bath size
            axis=1: input data time size
            axis=2: numbers of the nodes
            axis=3: output feature size
            '''

            encoder_init = encoder_lstm.lstm(self.para.batch_size * self.para.site_num,
                                             self.para.hidden_layer,
                                             self.para.hidden_size,
                                             self.para.is_training,
                                             placeholders=self.placeholders)
            inputs = tf.reshape(self.placeholders['features'],
                                shape=[self.para.batch_size, self.para.input_length, self.para.site_num,
                                       self.para.features])
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
            inputs = tf.reshape(inputs, shape=[self.para.batch_size * self.para.site_num, self.para.input_length,
                                               self.para.features])
            h_states, c_states = encoder_init.encoding(inputs)
            h_states = tf.reshape(h_states, shape=[self.para.batch_size, self.para.site_num, self.para.input_length,
                                                   self.para.hidden_size])
            h_states = tf.transpose(h_states, perm=[0, 2, 1, 3])

            print('h_states shape is : ', h_states.shape)

            # this step to presict the polutant concentration
            '''
            decoder, return --- for example ,output shape is :(32, 162, 1)
            axis=0: bath size
            axis=1: numbers of the nodes
            axis=2: label size
            '''
            decoder_init = decoder.lstm(self.para.batch_size * self.para.site_num,
                                        self.para.output_length,
                                        self.para.hidden_layer,
                                        self.para.hidden_size,
                                        placeholders=self.placeholders)

            inputs = tf.transpose(h_states, perm=[0, 2, 1, 3])
            inputs = tf.reshape(inputs, shape=[self.para.batch_size * self.para.site_num, self.para.input_length,
                                               self.para.hidden_size])
            pres = decoder_init.decoding(inputs)
            self.pres = tf.reshape(pres, shape=[self.para.batch_size, self.para.site_num, self.para.output_length])
            print('pres shape is : ', self.pres.shape)

            self.cross_entropy = tf.reduce_mean(
                tf.sqrt(tf.reduce_mean(tf.square(self.pres + 1e-10 - self.placeholders['labels']), axis=0)))

        print(self.cross_entropy)
        print('cross shape is : ', self.cross_entropy.shape)

        tf.summary.scalar('cross_entropy', self.cross_entropy)
        # backprocess and update the parameters
        self.train_op = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.cross_entropy)

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

    def accuracy(self, label, predict):
        '''
        :param label: represents the observed value
        :param predict: represents the predicted value
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

        # mask = label != 0
        # mape =np.mean(np.fabs((label[mask] - predict[mask]) / label[mask]))*100.0
        # mape=np.mean(np.fabs((label - predict) / label)) * 100.0
        # print('mape is: %.6f %' % (mape))
        sse = np.sum((label - predict) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        R2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
        print('r^2 is: %.6f' % (R2))

        return average_error, rmse_error, cor, R2

    def describe(self, label, predict, prediction_size):
        '''
        :param label:
        :param predict:
        :param prediction_size:
        :return:
        '''
        plt.figure()
        # Label is observed value,Blue
        plt.plot(label[0:prediction_size], 'b*:', label=u'actual value')
        # Predict is predicted value，Red
        plt.plot(predict[0:prediction_size], 'r*:', label=u'predicted value')
        # use the legend
        # plt.legend()
        plt.xlabel("time(hours)", fontsize=17)
        plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
        plt.title("the prediction of pm$_{2.5}", fontsize=17)
        plt.show()

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def re_current(self, a, max, min):
        return [num * (max - min) + min for num in a]

    def run_epoch(self):
        '''
        from now on,the model begin to training, until the epoch to 100
        '''

        max_rmse = 100
        self.sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        self.iterate = data_load.DataIterator(site_id=self.para.target_site_id,
                                              is_training=self.para.is_training,
                                              time_size=self.para.input_length,
                                              prediction_size=self.para.output_length,
                                              data_divide=self.para.data_divide,
                                              window_step=self.para.step,
                                              normalize=self.para.normalize,
                                              hp=self.para)

        iterate = self.iterate
        next_elements = iterate.next_batch(batch_size=self.para.batch_size, epochs=self.para.epochs, is_training=True)

        for i in range(int((iterate.length // self.para.site_num * iterate.data_divide - (
                iterate.time_size + iterate.prediction_size)) // iterate.window_step)
                       * self.para.epochs // self.para.batch_size):

            x, day, hour, label = self.sess.run(next_elements)
            features = np.reshape(x, [-1, self.para.site_num, self.para.features])
            day = np.reshape(day, [-1, self.para.site_num])
            hour = np.reshape(hour, [-1, self.para.site_num])
            feed_dict = construct_feed_dict(features, self.adj, label, day, hour, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.para.dropout})

            summary, loss, _ = self.sess.run((merged, self.cross_entropy, self.train_op), feed_dict=feed_dict)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss))
            writer.add_summary(summary, loss)

            # validate processing
            if i % 10 == 0:
                rmse_error = self.evaluate()

                if max_rmse > rmse_error:
                    print("the validate average rmse loss value is : %.6f" % (rmse_error))
                    max_rmse = rmse_error
                    self.saver.save(self.sess, save_path=self.para.save_path + 'model.ckpt')

                    # if os.path.exists('model_pb'): shutil.rmtree('model_pb')
                    # builder = tf.saved_model.builder.SavedModelBuilder('model_pb')
                    # builder.add_meta_graph_and_variables(self.sess, ["mytag"])
                    # builder.save()

    def evaluate(self):
        '''
        :param para:
        :param pre_model:
        :return:
        '''
        label_list = list()
        predict_list = list()

        # with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(self.para.save_path)
        if not self.para.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)

            self.saver.save(self.sess, save_path='gcn/model/' + 'model.ckpt')

        self.iterate_test = data_load.DataIterator(site_id=self.para.target_site_id,
                                                   is_training=self.para.is_training,
                                                   time_size=self.para.input_length,
                                                   prediction_size=self.para.output_length,
                                                   data_divide=self.para.data_divide,
                                                   normalize=self.para.normalize,
                                                   hp=self.para)
        iterate_test = self.iterate_test
        next_ = iterate_test.next_batch(batch_size=self.para.batch_size, epochs=1, is_training=False)
        max, min = iterate_test.max_list[-2], iterate_test.min_list[-2]

        # '''
        for i in range(int((iterate_test.length // self.para.site_num
                            - iterate_test.length // self.para.site_num * iterate_test.data_divide
                            - (
                                    iterate_test.time_size + iterate_test.prediction_size)) // iterate_test.prediction_size) // self.para.batch_size):
            x, day, hour, label = self.sess.run(next_)
            features = np.reshape(x, [-1, self.para.site_num, self.para.features])
            day = np.reshape(day, [-1, self.para.site_num])
            hour = np.reshape(hour, [-1, self.para.site_num])

            feed_dict = construct_feed_dict(features, self.adj, label, day, hour, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: 0.0})

            pre = self.sess.run((self.pres), feed_dict=feed_dict)
            label_list.append(label)
            predict_list.append(pre)

        label_list = np.reshape(np.array(label_list, dtype=np.float32),
                                [-1, self.para.site_num, self.para.output_length]).transpose([1, 0, 2])
        predict_list = np.reshape(np.array(predict_list, dtype=np.float32),
                                  [-1, self.para.site_num, self.para.output_length]).transpose([1, 0, 2])
        if self.para.normalize:
            label_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in label_list])
            predict_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in predict_list])
        else:
            label_list = np.array([np.reshape(site_label, [-1]) for site_label in label_list])
            predict_list = np.array([np.reshape(site_label, [-1]) for site_label in predict_list])

        np.savetxt('results/results_label.txt', label_list, '%.3f')
        np.savetxt('results/results_predict.txt', predict_list, '%.3f')

        label_list = np.reshape(label_list, [-1])
        predict_list = np.reshape(predict_list, [-1])
        average_error, rmse_error, cor, R2 = self.accuracy(label_list, predict_list)  # 产生预测指标
        # pre_model.describe(label_list, predict_list, pre_model.para.prediction_size)   #预测值可视化
        return average_error


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