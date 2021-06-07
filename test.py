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
import datetime
import argparse

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
logs_path="board"

class Model(object):
    def __init__(self,para):
        self.para=para
        self.adj=self.adjecent()
        self.iterate = data_load.DataIterator(site_id=self.para.target_site_id,
                                                is_training=True,
                                                time_size=self.para.input_length,
                                                prediction_size=self.para.output_length,
                                                data_divide=self.para.data_divide,
                                                window_step=self.para.step,
                                                normalize=self.para.normalize,
                                                hp=self.para)

        # self.x=tf.placeholder(dtype=tf.float32,shape=[None,self.para.input_length,self.para.features],name='inputs')
        # self.y=tf.placeholder(dtype=tf.float32,shape=[None,self.para.output_length],name='label')

        #define gcn model
        if self.para.model_name == 'gcn':
            self.support = [preprocess_adj(self.adj)]
            self.num_supports = 1
            self.model_func = GCN
        elif self.para.model_name == 'gcn_cheby':
            self.support = chebyshev_polynomials(self.adj, self.para.max_degree)
            self.num_supports = 1 + self.para.max_degree
            self.model_func = GCN
        else:
            raise ValueError('Invalid argument for model: ' + str(self.para.model_name))

        # define placeholders
        self.placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(self.num_supports)],
            # None : batch _size * time _size
            'features': tf.placeholder(tf.float32, shape=[self.para.batch_size*self.para.input_length, self.para.site_num, self.para.features]),
            'labels': tf.placeholder(tf.float32, shape=[self.para.batch_size, self.para.site_num, self.para.output_length]),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }
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
        normal=normalization.Normalization(inputs=self.placeholders['features'],out_size=self.para.features,is_training=True)
        normal.normal()

        # create model
        '''
        return, the gcn output --- for example, inputs.shape is :  (32, 3, 162, 32)
        axis=0: bath size
        axis=1: input data time size
        axis=2: numbers of the nodes
        axis=3: output feature size
        '''
        model_gcn = self.model_func(self.placeholders, input_dim=self.para.features, para=self.para)
        model_gcn_=[]
        for time in range(self.para.batch_size*self.para.input_length):
            model_gcn_.append(tf.expand_dims(model_gcn.predict(self.placeholders['features'][time]),axis=0))
        inputs=tf.concat(values=model_gcn_,axis=0)
        inputs=tf.reshape(inputs,shape=[self.para.batch_size,self.para.input_length,self.para.site_num,self.para.gcn_output_size])

        print('inputs shape is : ',inputs.shape)

        #this step use to encoding the input series data
        '''
        rlstm, return --- for example ,output shape is :(32, 162, 128)
        axis=0: bath size
        axis=1: numbers of the nodes
        axis=2: output features size
        '''
        encoder_init=encoder.encoder(self.para.hidden_layer,
                                     self.para.hidden_size,
                                     self.para.is_training)
        h_states=[]
        for site_id in range(inputs.shape[2]):
            (c_state, h_state)=encoder_init.encoding(inputs[:,:,site_id,:],
                                                     self.para.batch_size)
            h_states.append(tf.expand_dims(h_state,axis=0))
        h_states=tf.transpose(tf.concat(h_states,axis=0),perm=[1,0,2])

        print('h_states shape is : ', h_states.shape)

        # encoder_init=encoder_lstm.lstm(self.x,
        #                                self.para.batch_size,
        #                                self.para.hidden_layer,
        #                                self.para.hidden_size,
        #                                self.para.is_training)
        #encoder_init=encodet_gru.gru(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        # encoder_init=encoder_rnn.rnn(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        # h_state=encoder_init.encoding()

        #this step to presict the polutant concentration
        '''
        decoder, return --- for example ,output shape is :(32, 162, 1)
        axis=0: bath size
        axis=1: numbers of the nodes
        axis=2: label size
        '''
        decoder_init=decoder.lstm(self.para.batch_size,
                                  self.para.output_length,
                                  self.para.hidden_layer,
                                  self.para.hidden_size,
                                  self.para.is_training)
        self.pres=[]
        for site_id in range(h_states.shape[1]):
            pre=decoder_init.decoding(h_states[:,site_id,:])
            self.pres.append(tf.expand_dims(pre,axis=0))
        self.pres=tf.transpose(tf.concat(self.pres,axis=0),perm=[1,0,2])

        print('self.pres shape is : ', self.pres.shape)
        print('labels shape is : ', self.placeholders['labels'].shape)

        self.cross_entropy = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(self.placeholders['labels'] - self.pres), axis=0)))
        print(self.cross_entropy)
        print('cross shape is : ',self.cross_entropy.shape)

        tf.summary.scalar('cross_entropy',self.cross_entropy)
        # backprocess and update the parameters
        self.train_op = tf.train.AdamOptimizer(self.para.learning_rate).minimize(self.cross_entropy)

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

    def accuracy(self,label,predict):
        '''
        :param Label: represents the observed value
        :param Predict: represents the predicted value
        :param epoch:
        :param steps:
        :return:
        '''
        error = label - predict
        average_error = np.mean(np.fabs(error.astype(float)))
        print("mae is : %.6f" % (average_error))

        rmse_error = np.sqrt(np.mean(np.square(np.array(label) - np.array(predict))))
        print("rmse is : %.6f" % (rmse_error))

        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (predict - np.mean(predict)))) / (np.std(predict) * np.std(label))
        print('correlation coefficient is: %.6f' % (cor))
        return average_error,rmse_error,cor

    def describe(self,label,predict,prediction_size):
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
        self.sess=tf.Session()
        self.saver=tf.train.Saver()

    def re_current(self, a, max, min):
        return [round(float(num*(max-min)+min),3) for num in a]

    def run_epoch(self):
        '''
        from now on,the model begin to training, until the epoch to 100
        '''

        max_rmse = 100
        self.sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
        start_time = datetime.datetime.now()

        iterate=self.iterate
        next_elements=iterate.next_batch(batch_size=self.para.batch_size,epochs=self.para.epochs,is_training=True)

        # '''
        for i in range(int((iterate.length //self.para.site_num * iterate.data_divide-(iterate.time_size + iterate.prediction_size))//iterate.window_step)
                       * self.para.epochs // self.para.batch_size):
            x, label =self.sess.run(next_elements)

            # Construct feed dictionary
            # features = sp.csr_matrix(x)
            # features = preprocess_features(features)
            features=np.reshape(np.array(x), [-1, self.para.site_num, self.para.features])

            feed_dict = construct_feed_dict(features, self.support, label, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.para.dropout})

            summary, loss, _ = self.sess.run((merged,self.cross_entropy,self.train_op), feed_dict=feed_dict)
            writer.add_summary(summary, loss)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss))

            # '''
            # validate processing
            if i % 10 == 0:
                rmse_error=self.evaluate()

                if max_rmse>rmse_error:
                    print("the validate average rmse loss value is : %.6f" % (rmse_error))
                    max_rmse=rmse_error
                    self.saver.save(self.sess,save_path='weights/pollutant.ckpt')
            # '''
        # '''

        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print("Total running times is : %f" % total_time.total_seconds())

    def evaluate(self):
        '''
        :param para:
        :param pre_model:
        :return:
        '''
        label_list = list()
        predict_list = list()

        #with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint('weights/')
        if not self.para.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)

        iterate_test =self.iterate
        next_ = iterate_test.next_batch(batch_size=self.para.batch_size, epochs=1,is_training=False)
        max,min=iterate_test.max_list[-1],iterate_test.min_list[-1]


        # '''
        for i in range(int((iterate_test.length // self.para.site_num
                            -iterate_test.length // self.para.site_num * iterate_test.data_divide
                            -(iterate_test.time_size + iterate_test.prediction_size))//iterate_test.prediction_size)// self.para.batch_size):
            x, label =self.sess.run(next_)

            # Construct feed dictionary
            # features = sp.csr_matrix(x)
            # features = preprocess_features(features)
            features=np.reshape(np.array(x), [-1, self.para.site_num, self.para.features])
            feed_dict = construct_feed_dict(features, self.support, label, self.placeholders)
            # feed_dict.update({self.placeholders['dropout']: self.para.dropout})

            pre = self.sess.run((self.pres), feed_dict=feed_dict)
            label_list.append(label)
            predict_list.append(pre)

        label_list=np.reshape(np.array(label_list,dtype=np.float32),[-1, self.para.site_num, self.para.output_length]).transpose([1,0,2])
        predict_list=np.reshape(np.array(predict_list,dtype=np.float32),[-1, self.para.site_num, self.para.output_length]).transpose([1,0,2])
        if self.para.normalize:
            label_list = np.array([self.re_current(np.reshape(site_label, [-1]),max,min) for site_label in label_list],dtype=np.int32)
            predict_list = np.array([self.re_current(np.reshape(site_label, [-1]),max,min) for site_label in predict_list],dtype=np.int32)
        else:
            label_list = np.array([np.reshape(site_label, [-1]) for site_label in label_list],dtype=np.int32)
            predict_list = np.array([np.reshape(site_label, [-1]) for site_label in predict_list],dtype=np.int32)

        np.savetxt('results/results_label.txt',label_list,'%.3f')
        np.savetxt('results/results_predict.txt', predict_list, '%.3f')

        label_list=np.reshape(label_list,[-1])
        predict_list=np.reshape(predict_list,[-1])
        average_error, rmse_error, cor = self.accuracy(label_list, predict_list)  #产生预测指标
        #pre_model.describe(label_list, predict_list, pre_model.para.prediction_size)   #预测值可视化
        return rmse_error

def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    print('beginning____________________________beginning_____________________________beginning!!!')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')
    pre_model = Model(para)
    pre_model.initialize_session()

    if int(val) == 1:para.is_training = True
    else:
        para.batch_size=32
        para.is_training = False

    if int(val) == 1:pre_model.run_epoch()
    else:
        pre_model.evaluate()

    print('finished____________________________finished_____________________________finished!!!')

if __name__ == '__main__':
    main()