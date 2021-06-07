# -- coding: utf-8 --

import tensorflow as tf
import numpy as np
import model.decoder as decoder
import matplotlib.pyplot as plt
import model.normalization as normalization
import model.encoder as encoder
import model.encoder_lstm as encoder_lstm
import model.data_process as data_load
from model.hyparameter import parameter
import os
import datetime
import argparse

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
logs_path="board"

class Model(object):
    def __init__(self,para):
        self.para=para
        self.iterate = data_load.DataIterator(site_id=self.para.target_site_id,
                                                is_training=True,
                                                time_size=self.para.input_length,
                                                prediction_size=self.para.output_length,
                                                data_divide=self.para.data_divide,
                                                window_step=self.para.step,
                                                normalize=self.para.normalize)

        self.x=tf.placeholder(dtype=tf.float32,shape=[None,self.para.input_length,self.para.features],name='inputs')
        self.y=tf.placeholder(dtype=tf.float32,shape=[None,self.para.output_length],name='label')
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

        '''
        feedforward and BN layer
        output shape:[batch, time_size,field_size,new_features]
        '''
        normal=normalization.Normalization(inputs=self.x,out_size=self.para.features,is_training=True)
        normal.normal()
        #this step use to encoding the input series data


        encoder_init=encoder.encoder(self.para.hidden_layer,
                                     self.para.hidden_size,
                                     self.para.is_training)
        (c_state, h_state)=encoder_init.encoding(self.x,
                                     self.para.batch_size)
        #
        # encoder_init=encoder_lstm.lstm(self.x,
        #                                self.para.batch_size,
        #                                self.para.hidden_layer,
        #                                self.para.hidden_size,
        #                                self.para.is_training)
        #encoder_init=encodet_gru.gru(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        # encoder_init=encoder_rnn.rnn(self.x_input,batch_size,encoder_layer,encoder_nodes,is_training)
        # h_state=encoder_init.encoding()

        #this step to presict the polutant concentration
        decoder_init=decoder.lstm(self.para.batch_size,
                                  self.para.output_length,
                                  self.para.hidden_layer,
                                  self.para.hidden_size,
                                  self.para.is_training)
        self.pre=decoder_init.decoding(h_state)

        self.cross_entropy = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(self.y - self.pre), axis=0)), axis=0)
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

        #return

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

    def re_current(self, a,max,min):
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

        for i in range((int(iterate.length * iterate.data_divide) -
                        (iterate.time_size + iterate.prediction_size))*self.para.epochs
                       // (iterate.window_step*self.para.batch_size)):
            x, label =self.sess.run(next_elements)
            summary, loss, _ ,t= self.sess.run((merged,self.cross_entropy,self.train_op,self.pre), feed_dict={self.x: x, self.y: label})
            writer.add_summary(summary, loss)

            # validate processing
            rmse_error=self.evaluate()
            print("After %d steps,the validate average rmse loss value is : %.6f:" % (i, loss))
            if max_rmse>rmse_error:
                max_rmse=rmse_error
                self.saver.save(self.sess,save_path='weights/pollutant.ckpt')

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

        for i in range((int(iterate_test.length * (1-iterate_test.data_divide)) - iterate_test.prediction_size)//
                       (iterate_test.time_size * self.para.batch_size)):
            x, label = self.sess.run(next_)
            pre= self.sess.run((self.pre),feed_dict={self.x: x})
            label_list.append(label)
            predict_list.append(pre)
        if self.para.normalize:
            label_list = np.array(self.re_current(np.reshape(np.array(label_list), [-1]),max,min))
            predict_list = np.array(self.re_current(np.reshape(np.array(predict_list), [-1]),max,min))
        else:
            label_list = np.reshape(np.array(label_list,dtype=np.int32), [-1])
            predict_list = np.reshape(np.array(predict_list,dtype=np.int32), [-1])

        np.savetxt('results/results.txt',(label_list,predict_list),'%.3f')

        average_error, rmse_error, cor = self.accuracy(label_list, predict_list)  #产生预测指标
        #pre_model.describe(label_list, predict_list, pre_model.para.prediction_size)   #预测值可视化
        return rmse_error

def main(argv=None):
    '''
    :param argv:
    :return:
    '''
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