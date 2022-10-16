# -- coding: utf-8 --

import tensorflow as tf

class FirnnClass(object):
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
            lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
            lstm_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1.0)
            return lstm_cell_
        self.e_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        self.e_initial_state = self.e_mlstm.zero_state(self.batch_size, tf.float32)

    def decoder(self):
        def cell():
            lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
            lstm_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1-self.placeholders['dropout'])
            return lstm_cell_
        self.d_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        self.d_initial_state = self.d_mlstm.zero_state(self.batch_size, tf.float32)

    def encoding(self, inputs, STE=None):
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''
        # out put the store data
        with tf.variable_scope('encoder_lstm'):
            ouputs, state = tf.nn.dynamic_rnn(cell=self.e_mlstm, inputs=inputs,initial_state=self.e_initial_state,dtype=tf.float32)

        a = tf.layers.dense(inputs=STE, units=64,kernel_initializer=tf.initializers.truncated_normal(),activation=tf.nn.sigmoid)
        a = tf.layers.dense(inputs=a, units=1, kernel_initializer=tf.initializers.truncated_normal(),
                            activation=tf.nn.sigmoid)
        a_score = tf.nn.softmax(a) # [batch * site num, input times, 1]
        a_score = tf.transpose(a_score, perm=[0, 2, 1]) # [batch * site num, 1, input times]
        outputs = tf.matmul(a_score, ouputs)

        return outputs # [batch size, 1, hidden size]

    def decoding_(self,  encoder_hs, site_num):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''
        x = tf.squeeze(encoder_hs, axis=1)
        x = tf.reshape(x, shape=[-1, site_num, x.shape[-1]])
        results_speed = tf.layers.dense(inputs=x, units=64, activation=tf.nn.relu, name='layer_spped_1',
                                        reuse=tf.AUTO_REUSE)
        results = tf.layers.dense(inputs=results_speed, units=self.predict_time, activation=tf.nn.relu,
                                  name='layer_speed_2', reuse=tf.AUTO_REUSE)

        return results

    def decoding(self,  encoder_hs, site_num):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''

        pres = []
        h_state = encoder_hs[:, -1, :]
        initial_state=self.d_initial_state

        for i in range(self.predict_time):
            h_state = tf.expand_dims(input=h_state, axis=1)

            with tf.variable_scope('decoder_lstm'):
                h_state, state = tf.nn.dynamic_rnn(cell=self.d_mlstm, inputs=h_state,
                                                   initial_state=initial_state, dtype=tf.float32)
                initial_state = state

            h_state=tf.reshape(h_state,shape=[-1,self.nodes])

            results = tf.layers.dense(inputs=h_state, units=1, name='layer', reuse=tf.AUTO_REUSE)
            pre=tf.reshape(results,shape=[-1,site_num])
            # to store the prediction results for road nodes on each time
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