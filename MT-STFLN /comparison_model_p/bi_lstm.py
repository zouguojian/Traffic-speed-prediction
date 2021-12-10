# -- coding: utf-8 --

import tensorflow as tf

class b_lstm(object):
    def __init__(self, batch_size, predict_time=1, layer_num=1, nodes=128, placeholders=None):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.predict_time=predict_time
        self.placeholders=placeholders
        self.encoder()
        self.decoder()

    def encoder(self):
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

        self.ef_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell_fw for _ in range(self.layer_num)])
        self.eb_mlstm = tf.nn.rnn_cell.MultiRNNCell([cell_bw for _ in range(self.layer_num)])

    def decoder(self):
        '''
        :return:
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

    def encoding(self, inputs):
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''

        with tf.variable_scope('encoder_lstm'):

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.ef_mlstm, self.eb_mlstm, inputs,
                                                                        dtype=tf.float32)  # [2, batch_size, seq_length, output_size]
            outputs = tf.concat(outputs, axis=2)

        return outputs

    def decoding(self,  encoder_hs):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''

        pres = []
        h_state = encoder_hs[:, -1:, :]

        for i in range(self.predict_time):

            with tf.variable_scope('decoder_lstm'):
                h_state, _ = tf.nn.bidirectional_dynamic_rnn(self.df_mlstm, self.db_mlstm, h_state,
                                                             dtype=tf.float32)

                h_state = tf.concat(h_state, axis=2)

            results = tf.layers.dense(inputs=tf.squeeze(h_state,axis=1), units=1, name='layer', reuse=tf.AUTO_REUSE)
            # to store the prediction results for road nodes on each time
            pres.append(results)

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