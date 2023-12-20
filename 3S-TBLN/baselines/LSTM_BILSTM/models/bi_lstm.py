# -- coding: utf-8 --
from models.inits import *

class BilstmClass(object):
    def __init__(self, hp, placeholders=None):
        '''
        :param hp:
        :param placeholders:
        '''
        self.hp = hp
        self.batch_size = self.hp.batch_size
        self.layer_num = self.hp.hidden_layer
        self.hidden_size = self.hp.hidden_size
        self.input_length = self.hp.input_length
        self.output_length = self.hp.output_length
        self.placeholders = placeholders
        self.encoder()
        self.decoder()

    def lstm(self):
        def cell():
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)
            lstm_cell_ = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1-self.placeholders['dropout'])
            return lstm_cell_
        mlstm = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        return mlstm

    def bilstm(self):
        def cell():
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)  # single lstm unit
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=1-self.placeholders['dropout'])
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)  # single lstm unit
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=1-self.placeholders['dropout'])
            return cell_fw, cell_bw
        cell_fw, cell_bw=cell()
        f_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell_fw for _ in range(self.layer_num)])
        b_mlstm = tf.nn.rnn_cell.MultiRNNCell([cell_bw for _ in range(self.layer_num)])
        return f_mlstm, b_mlstm

    def encoder(self):
        '''
        :return:  shape is [batch size, time size, hidden size]
        '''
        self.e_lstm_1 = self.lstm()
        self.ef_bilstm_2, self.eb_bilstm_2 = self.bilstm()
        self.e_lstm_3 = self.lstm()

    def decoder(self):
        '''
        :return:
        '''
        self.d_lstm_1 = self.lstm()
        self.df_bilstm_2, self.db_bilstm_2 = self.bilstm()
        self.d_lstm_3 = self.lstm()

    def encoding(self, inputs):
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''
        with tf.variable_scope('encoder_lstm_1'):
            lstm_1_outpus, _ = tf.nn.dynamic_rnn(cell=self.e_lstm_1, inputs=inputs, dtype=tf.float32)
            x = lstm_1_outpus
        with tf.variable_scope('encoder_bilstm_2'):
            bilstm_2_outpus, _ = tf.nn.bidirectional_dynamic_rnn(self.ef_bilstm_2, self.eb_bilstm_2, x, dtype=tf.float32)
            # shape is [2, batch_size, seq_length, output_size]
            x = tf.concat(bilstm_2_outpus, axis=2)
            x = tf.layers.dense(inputs=x, units=self.hidden_size, activation=None,name='encoder_full')
        with tf.variable_scope('encoder_lstm_3'):
            lstm_3_outpus,_ = tf.nn.dynamic_rnn(cell=self.e_lstm_3, inputs=x, dtype=tf.float32)
            x = lstm_3_outpus
        return x

    def decoding(self,  encoder_hs, site_num):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''
        pres = []
        h_state = encoder_hs[:, -1:, :]
        for i in range(self.output_length):
            with tf.variable_scope('decoder_lstm_1'):
                lstm_1_outpus, _ = tf.nn.dynamic_rnn(cell=self.d_lstm_1, inputs=h_state, dtype=tf.float32)
                x = lstm_1_outpus
            with tf.variable_scope('decoder_bilstm_2'):
                bilstm_2_outpus, _ = tf.nn.bidirectional_dynamic_rnn(self.df_bilstm_2, self.db_bilstm_2, x, dtype=tf.float32)
                # shape is [2, batch_size, seq_length, output_size]
                x = tf.concat(bilstm_2_outpus, axis=2)
            with tf.variable_scope('decoder_lstm_3'):
                lstm_3_outpus,_ = tf.nn.dynamic_rnn(cell=self.d_lstm_3, inputs=x, dtype=tf.float32)
            h_state = lstm_3_outpus
            layer_1 = tf.layers.dense(inputs=tf.squeeze(h_state), units=64, name='layer1', reuse=tf.AUTO_REUSE)
            results = tf.layers.dense(inputs=layer_1, units=1, name='layer2', reuse=tf.AUTO_REUSE)
            pre = tf.reshape(results, shape=[-1, site_num])
            pres.append(tf.expand_dims(pre, axis=-1))

        return tf.concat(pres, axis=-1,name='output_y')

import numpy as np
if __name__ == '__main__':
    train_data=np.random.random(size=[32,3,16])
    x=tf.placeholder(tf.float32, shape=[32, 3, 16])
    r=lstm(32,10,2,128)
    hs=r.encoding(x)

    print(hs.shape)

    pre=r.decoding(hs)
    print(pre.shape)