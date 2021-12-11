# -- coding: utf-8 --

from __future__ import division
from __future__ import print_function
from model.hyparameter import parameter
from model.gan import embedding
from model.gan import Transformer
from model.utils import construct_feed_dict

import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
import model.decoder as decoder
import matplotlib.pyplot as plt
import model.lstm as encoder_lstm
import model.data_load as data_load
import os
import argparse
import shutil

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"


class Model(object):
    def __init__(self, para):
        self.para = para

        # define placeholders
        self.placeholders = {
            'position': tf.placeholder(tf.int32, shape=(1, self.para.site_num), name='input_position'),
            'day': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_day'),
            'hour': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_hour'),
            'indices_i': tf.placeholder(dtype=tf.int64, shape=[None, None], name='input_indices'),
            'values_i': tf.placeholder(dtype=tf.float32, shape=[None], name='input_values'),
            'dense_shape_i': tf.placeholder(dtype=tf.int64, shape=[None], name='input_dense_shape'),
            # None: batch size * time size
            'features': tf.placeholder(tf.float32, shape=[None, self.para.site_num, self.para.features],
                                       name='input_features'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.para.site_num, self.para.output_length],
                                     name='labels'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='input_zero')  # helper variable for sparse dropout
        }

        self.model()

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

        with tf.variable_scope('position'):
            p_emd = embedding(self.placeholders['position'], vocab_size=self.para.site_num,
                              num_units=self.para.emb_size,
                              scale=False, scope="position_embed")
            p_emd = tf.reshape(p_emd, shape=[1, self.para.site_num, self.para.emb_size])
            p_emd = tf.expand_dims(p_emd, axis=0)
            self.p_emd = tf.tile(p_emd, [self.para.batch_size, self.para.input_length + self.para.output_length, 1, 1])
            print('p_emd shape is : ', self.p_emd.shape)

        with tf.variable_scope('day'):
            self.d_emb = embedding(self.placeholders['day'], vocab_size=32, num_units=self.para.emb_size,
                                   scale=False, scope="day_embed")
            self.d_emd = tf.reshape(self.d_emb,
                                    shape=[self.para.batch_size, self.para.input_length + self.para.output_length,
                                           self.para.site_num, self.para.emb_size])
            print('d_emd shape is : ', self.d_emd.shape)

        with tf.variable_scope('hour'):
            self.h_emb = embedding(self.placeholders['hour'], vocab_size=24, num_units=self.para.emb_size,
                                   scale=False, scope="hour_embed")
            self.h_emd = tf.reshape(self.h_emb,
                                    shape=[self.para.batch_size, self.para.input_length + self.para.output_length,
                                           self.para.site_num, self.para.emb_size])
            print('h_emd shape is : ', self.h_emd.shape)

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
            features = tf.layers.dense(self.placeholders['features'],
                                       units=self.para.emb_size)  # [-1, site num, emb_size]
            in_day = self.d_emd[:, :self.para.input_length, :, :]
            in_hour = self.h_emd[:, :self.para.input_length, :, :]
            in_position=self.p_emd[:, :self.para.input_length, :, :]

            # transformer
            m = Transformer(self.para)
            x = m.encoder(speed=features, day=in_day, hour=in_hour, position=self.p_emd[:, :self.para.input_length, :, :])

            # features=tf.add_n(inputs=[features,
            #                           tf.reshape(in_day,[-1,self.para.site_num, self.para.emb_size]),
            #                           tf.reshape(in_hour,[-1,self.para.site_num, self.para.emb_size])])

            print('encoder gan outs shape is : ', x.shape)

            # this step use to encoding the input series data
            '''
            lstm, return --- for example ,output shape is :(32, 3, 162, 128)
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

            x = tf.reshape(x, shape=[self.para.batch_size, self.para.input_length, self.para.site_num,
                                     self.para.emb_size])
            # trick
            inputs = tf.add_n([x, in_position])

            # inputs=tf.add_n([inputs, x_p])
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
            inputs = tf.reshape(inputs, shape=[self.para.batch_size * self.para.site_num, self.para.input_length,
                                               self.para.gcn_output_size])

            # encoder_outs = tf.transpose(x, perm=[0, 2, 1, 3])# 注意修改
            # encoder_outs=tf.reshape(encoder_outs, shape=[self.para.batch_size * self.para.site_num, self.para.input_length,
            #                                               self.para.gcn_output_size])# 注意修改
            h_states, c_states = encoder_init.encoding(inputs)  # 注意修改
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
            out_position=self.p_emd[:, self.para.input_length:, :, :]

            # transformer
            decoder_m = Transformer(self.para)

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

        self.pres = decoder_init.gcn_decoding(h_states,
                                               gan=decoder_m,
                                               site_num=self.para.site_num,
                                               day=out_day,
                                               hour=out_hour,
                                               position=out_position)

        print('pres shape is : ', self.pres.shape)

        self.loss = tf.reduce_mean(
            tf.sqrt(tf.reduce_mean(tf.square(self.pres + 1e-10 - self.placeholders['labels']), axis=0)))
            # weights=tf.Variable(initial_value=tf.constant(value=0.5, dtype=tf.float32),name='loss_weight')

            # self.cross_entropy = loss1 + loss2
        # backprocess and update the parameters
        # self.train_op = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.cross_entropy)
        self.train_op_1 = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.loss)

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

            x, day, hour, label= self.sess.run(next_elements)
            features = np.reshape(x, [-1, self.para.site_num, self.para.features])
            day = np.reshape(day, [-1, self.para.site_num])
            hour = np.reshape(hour, [-1, self.para.site_num])
            feed_dict = construct_feed_dict(features, label, day, hour, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.para.dropout})

            # summary, loss, _ = self.sess.run((merged, self.cross_entropy, self.train_op), feed_dict=feed_dict)

            loss, _ = self.sess.run((self.loss, self.train_op_1), feed_dict=feed_dict)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss))
            # writer.add_summary(summary, loss)

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

            feed_dict = construct_feed_dict(features, label, day, hour, self.placeholders)
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

        label_list = np.reshape(label_list, [-1])
        predict_list = np.reshape(predict_list, [-1])
        average_error, rmse_error, cor, R2 = self.accuracy(label_list, predict_list)  # 产生预测指标
        # self.describe(label_list, predict_list)   #预测值可视化
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