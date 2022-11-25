# -- coding: utf-8 --

import tensorflow as tf

class LstmClass(object):
    def __init__(self, batch_size, layer_num=1, nodes=128, placeholders=None):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.placeholders = placeholders
        self.encoder()

    def lstm_cell(self):
        '''
        :return:
        '''
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
        return tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=1-self.placeholders['dropout'])

    def encoder(self):
        '''
        :return:
        '''
        self.mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        self.initial_state=self.mlstm_cell.zero_state(self.batch_size,tf.float32)

    def encoding(self, inputs=None):
        '''
        :param inputs:
        :return:
        '''
        initial_state=self.initial_state
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            h_states, c_states = tf.nn.dynamic_rnn(cell=self.mlstm_cell, inputs=inputs, initial_state=initial_state, dtype=tf.float32)
        return (h_states,c_states)