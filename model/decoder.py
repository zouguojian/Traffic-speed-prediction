# -- coding: utf-8 --

import tensorflow as tf
class lstm(object):
    def __init__(self,batch_size,predict_time,layer_num=1,nodes=128,is_training=True):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.is_training=is_training
        self.keep_pro()
        self.predict_time=predict_time

    def keep_pro(self):
        '''
        used to define the self.keepProb value
        :return:
        '''
        if self.is_training:self.keepProb=0.5
        else:self.keepProb=1.0

    def lstm_cell(self):
        '''
        :return: lstm
        '''
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.nodes,reuse=tf.AUTO_REUSE)
        return tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keepProb)

    def attention(self, h_t, encoder_hs):
        '''
        h_t for decoder, the shape is [batch, h]
        encoder_hs for encoder, the shape is [batch, time ,h]
        :param h_t:
        :param encoder_hs:
        :return:
        '''
        #scores = [tf.matmul(tf.tanh(tf.matmul(tf.concat(1, [h_t, tf.squeeze(h_s, [0])]),
        #                    self.W_a) + self.b_a), self.v_a)
        #          for h_s in tf.split(0, self.max_size, encoder_hs)]
        #scores = tf.squeeze(tf.pack(scores), [2])
        scores = tf.reduce_sum(tf.multiply(encoder_hs, tf.tile(h_t,multiples=[1,encoder_hs.shape[1],1])), 2)
        # a_t    = tf.nn.softmax(tf.transpose(scores))  #[batch, time]
        a_t = tf.nn.softmax(scores)  # [batch, time]
        a_t    = tf.expand_dims(a_t, 2) #[batch, time ,1]
        c_t    = tf.matmul(tf.transpose(encoder_hs, perm=[0,2,1]), a_t) #[batch ,h , 1]
        c_t    = tf.squeeze(c_t) #[batch, h]]
        h_t=tf.squeeze(h_t)
        h_tld  = tf.layers.dense(tf.concat([h_t, c_t], axis=1),units=c_t.shape[-1],activation=tf.nn.relu) #[batch, h]
        # print('h_tld shape is : ',h_tld.shape)
        return h_tld

    def decoding(self,encoder_hs):
        '''
        :param h_state:
        :return:
        '''

        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        # print('h_state shape is : ',h_state.shape)
        self.initial_state=mlstm_cell.zero_state(self.batch_size,tf.float32)
        h=[]

        h_state=encoder_hs[:,-1,:]
        for i in range(self.predict_time):
            h_state = tf.expand_dims(input=h_state,axis=1)

            with tf.variable_scope('decoder_lstm', reuse=tf.AUTO_REUSE):
                h_state, state = tf.nn.dynamic_rnn(cell=mlstm_cell, inputs=h_state,initial_state=self.initial_state,dtype=tf.float32)
                # h_state=tf.squeeze(h_state)
                h_state=self.attention(h_t=h_state,encoder_hs=encoder_hs) #attention
                self.initial_state=state

                results=tf.layers.dense(inputs=h_state,units=1,name='layer',reuse=tf.AUTO_REUSE)
            h.append(results)

        return tf.squeeze(tf.transpose(tf.convert_to_tensor(h),[1,2,0]),axis=1)