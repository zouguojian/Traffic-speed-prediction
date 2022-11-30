# -- coding: utf-8 --

from __future__ import division
from __future__ import print_function

import pandas as pd
import os
from baseline.tgcn.tgcn import tgcnCell
from baseline.tgcn.utils import *
from baseline.tgcn.hyparameter import parameter
import matplotlib.pyplot as plt
import baseline.tgcn.data_next as data_load
import argparse
import datetime
import csv

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"

# os.environ['CUDA_VISIBLE_DEVICES']='2'
#
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


class Model(object):
    def __init__(self, hp):
        '''
        :param para:
        '''
        self.hp = hp             # hyperparameter
        self.init_placeholder()  # init placeholder
        self.model()             # init prediction model


    def init_placeholder(self):
        '''
        :return:
        '''
        self.placeholders = {
            'features': tf.placeholder(tf.float32, shape=[None, self.hp.input_length, self.hp.site_num], name='input_features'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.hp.site_num, self.hp.output_length], name='labels'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout')
        }

    def adjecent(self):
        '''
        :return: adjacent matrix
        '''
        data = pd.read_csv(filepath_or_buffer=self.hp.file_adj)
        adj = np.zeros(shape=[self.hp.site_num, self.hp.site_num])
        for line in data[['src_FID', 'nbr_FID']].values:
            adj[line[0]][line[1]] = 1
        return adj

    def model(self):
        '''
        :return:
        '''

        def TGCN(_X, adj):
            ###
            cell_1 = tgcnCell(self.hp.hidden_size, adj, num_nodes=self.hp.site_num)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)  # 可用多层
            _X = tf.unstack(_X, axis=1)
            outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
            print('outputs length is : ', len(outputs))
            print('outputs shape is : ', outputs[-1].shape)
            m = []
            for i in outputs:
                o = tf.reshape(i, shape=[-1, self.hp.site_num, self.hp.hidden_size])
                o = tf.reshape(o, shape=[-1, self.hp.hidden_size])
                m.append(o)
            last_output = m[-1]
            print('last_output shape is : ', last_output.shape)
            last_output = tf.reshape(last_output, [-1, self.hp.site_num, self.hp.hidden_size])
            last_output = tf.reshape(last_output, [-1, self.hp.site_num, self.hp.hidden_size])
            last_output = tf.layers.dense(inputs=last_output, units=64, activation=tf.nn.relu, name='layer_1')
            output = tf.layers.dense(inputs=last_output, units=self.hp.output_length, activation=tf.nn.relu, name='output_y')
            # output = tf.layers.dense(last_output,units=self.hp.output_length,name='output_y')
            return output, m, states

        adj = self.adjecent()

        self.pre, _, _ = TGCN(self.placeholders['features'],adj=adj)

        print('pres shape is : ', self.pre.shape)

        self.loss = tf.reduce_mean(
                tf.sqrt(tf.reduce_mean(tf.square(self.pre + 1e-10 - self.placeholders['labels']), axis=0)))
        self.train_op = tf.train.AdamOptimizer(self.hp.learning_rate).minimize(self.loss)

    def test(self):
        '''
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def describe(self, label, predict):
        '''
        :param label:
        :param predict:
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
        :return:
        '''
        max_mae = 100
        self.sess.run(tf.global_variables_initializer())

        iterate = data_load.DataClass(hp=self.hp)
        train_next = iterate.next_batch(batch_size=self.hp.batch_size, epoch=self.hp.epoch, is_training=True)

        for i in range(int((iterate.length // self.hp.site_num * iterate.divide_ratio - (
                iterate.input_length + iterate.output_length)) // iterate.step)
                       * self.hp.epoch // self.hp.batch_size):
            x, label,_,_,_ = self.sess.run(train_next)
            features = np.reshape(x, [-1, self.hp.input_length, self.hp.site_num])
            feed_dict = construct_feed_dict(features,label, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: self.hp.dropout})
            loss_, _ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
            print("after %d steps,the training average loss value is : %.6f" % (i, loss_))

            # validate processing
            if i % 100 == 0:
                mae = self.evaluate()

                if max_mae > mae:
                    print("the validate average rmse loss value is : %.6f" % (mae))
                    max_mae = mae
                    self.saver.save(self.sess, save_path=self.hp.save_path + 'model.ckpt')

                    # if os.path.exists('model_pb'): shutil.rmtree('model_pb')
                    # builder = tf.saved_model.builder.SavedModelBuilder('model_pb')
                    # builder.add_meta_graph_and_variables(self.sess, ["mytag"])
                    # builder.save()

    def evaluate(self):
        '''
        :return:
        '''
        label_list = list()
        predict_list = list()

        # with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint(self.hp.save_path)
        if not self.hp.is_training:
            print('the model weights has been loaded:')
            self.saver.restore(self.sess, model_file)
            # self.saver.save(self.sess, save_path='gcn/model/' + 'model.ckpt')

        iterate_test = data_load.DataClass(hp=self.hp)
        test_next = iterate_test.next_batch(batch_size=self.hp.batch_size, epoch=1, is_training=False)
        max, min = iterate_test.max_dict['speed'], iterate_test.min_dict['speed']
        print(max, min)

        file = open('results/' + str(self.hp.model_name) + '.csv', 'w', encoding='utf-8')
        writer = csv.writer(file)
        writer.writerow(
            ['road'] + ['day_' + str(i) for i in range(self.hp.output_length)] + ['hour_' + str(i) for i in range(
                self.hp.output_length)] +
            ['minute_' + str(i) for i in range(self.hp.output_length)] + ['label_' + str(i) for i in
                                                                            range(self.hp.output_length)] +
            ['predict_' + str(i) for i in range(self.hp.output_length)])

        # '''
        for i in range(int((iterate_test.length // self.hp.site_num
                            - iterate_test.length // self.hp.site_num * iterate_test.divide_ratio
                            - (iterate_test.input_length + iterate_test.output_length)) // iterate_test.output_length)
                       // self.hp.batch_size):
            x, label, day, hour, minute = self.sess.run(test_next)
            features = np.reshape(x, [-1, self.hp.input_length, self.hp.site_num])
            feed_dict = construct_feed_dict(features, label, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: 0.0})

            # if i==0: begin_time = datetime.datetime.now()
            pre = self.sess.run((self.pre), feed_dict=feed_dict)

            day = np.reshape(day, [-1, self.hp.site_num])
            hour = np.reshape(hour, [-1, self.hp.site_num])
            minute = np.reshape(minute, [-1, self.hp.site_num])

            for site in range(self.hp.site_num):
                writer.writerow([site]+list(day[self.hp.input_length:,0])+
                                 list(hour[self.hp.input_length:,0])+
                                 list(minute[self.hp.input_length:,0]*15)+
                                 list(np.round(self.re_current(label[0][site],max,min)))+
                                 list(np.round(self.re_current(pre[0][site],max,min))))

            # if i == 0:
            #     end_t = datetime.datetime.now()
            #     total_t = end_t - begin_time
            #     print("Total running times is : %f" % total_t.total_seconds())

            label_list.append(label)
            predict_list.append(pre)

        label_list = np.reshape(np.array(label_list, dtype=np.float32),
                                [-1, self.hp.site_num, self.hp.predict_length]).transpose([1, 0, 2])
        predict_list = np.reshape(np.array(predict_list, dtype=np.float32),
                                  [-1, self.hp.site_num, self.hp.predict_length]).transpose([1, 0, 2])

        if self.hp.normalize:
            label_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in label_list])
            predict_list = np.array(
                [self.re_current(np.reshape(site_label, [-1]), max, min) for site_label in predict_list])
        else:
            label_list = np.array([np.reshape(site_label, [-1]) for site_label in label_list])
            predict_list = np.array([np.reshape(site_label, [-1]) for site_label in predict_list])

        label_all = np.reshape(np.array(label_list),newshape=[self.hp.site_num, -1, self.hp.output_length])
        predict_all = np.reshape(np.array(predict_list), newshape=[self.hp.site_num, -1, self.hp.output_length])

        label_list = np.reshape(label_list, [-1])
        predict_list = np.reshape(predict_list, [-1])

        # average_error, rmse_error, cor, R2 = accuracy(label_list, predict_list)  # 产生预测指标
        mae, rmse, mape, cor, r2=metric(predict_list,label_list)

        for i in range(self.hp.output_length):
            print('in the %d time step, the evaluating indicator'%(i+1))
            metric(np.reshape(predict_all[:,:,i], [-1]), np.reshape(label_all[:,:,i], [-1]))
        # self.describe(label_list, predict_list)   #预测值可视化
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