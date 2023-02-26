# -- coding: utf-8 --
import tensorflow as tf
from baseline.astgat.spatial_attention import Transformer

class AstGatClass(object):
    def __init__(self,hp, placeholders=None):
        self.hp = hp
        self.hidden_size = self.hp.hidden_size
        self.layer_num = self.hp.hidden_layer
        self.placeholders = placeholders

    def lstm(self):
        def cell():
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)
            lstm_cell_ = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1-self.placeholders['dropout'])
            return lstm_cell_
        mlstm = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        return mlstm

    def attention(self, queries, keys, num_units):
        # Linear projections
        Q = tf.layers.dense(queries, num_units, use_bias=False)  # (N, 1, h)
        K = tf.layers.dense(keys, num_units, use_bias=False)     # (N, 24, h)
        V = keys     # (N, 24, h)
        Q = tf.tile(Q,[1, self.hp.input_length*4, 1])

        bias = tf.Variable(tf.truncated_normal(shape=[num_units]), name='bias')

        QK = tf.nn.tanh(tf.add(tf.add(Q, K),bias))
        scores = tf.layers.dense(QK, units=1,use_bias=False)
        scores = tf.transpose(scores, [0, 2, 1])
        outputs = tf.nn.softmax(scores)

        outputs = tf.matmul(outputs, V)
        return outputs

    def attention_s(self, h_t, encoder_hs):
        '''
        h_t for decoder, the shape is [batch, 1, h]
        encoder_hs for encoder, the shape is [batch, time ,h]
        :param h_t:
        :param encoder_hs:
        :return: [None, hidden size]
        '''
        scores = tf.reduce_sum(tf.multiply(encoder_hs, tf.tile(h_t,multiples=[1,encoder_hs.shape[1],1])), 2)
        a_t = tf.nn.softmax(scores)  # [batch, time]
        a_t = tf.expand_dims(a_t, 2) # [batch, time, 1]
        c_t = tf.matmul(tf.transpose(encoder_hs, perm=[0,2,1]), a_t) #[batch ,h , 1]
        c_t = tf.squeeze(c_t, axis=2) #[batch, h]]
        h_t=tf.squeeze(h_t,axis=1)
        h_tld  = tf.layers.dense(tf.concat([h_t, c_t], axis=1),units=c_t.shape[-1],activation=tf.nn.relu) #[batch, h]
        return h_tld

    def encoder(self, x_w, x_d, x_r):
        '''
        :param x_w:  [batch, length, site, hidden]
        :param x_d:
        :param x:
        :return:
        '''
        with tf.variable_scope('spatial_1'):
            spatial_1=Transformer(self.hp)
            x_w = tf.reshape(x_w, [-1, self.hp.site_num, self.hp.features])
            x_w = spatial_1.encoder(inputs=x_w, input_length=self.hp.input_length*2)
            x_w = tf.reshape(x_w, [-1, self.hp.input_length * 2, self.hp.site_num, self.hp.emb_size])

        with tf.variable_scope('spatial_2'):
            spatial_2 = Transformer(self.hp)
            x_d = tf.reshape(x_d, [-1, self.hp.site_num, self.hp.features])
            x_d = spatial_2.encoder(inputs=x_d, input_length=self.hp.input_length*2)
            x_d = tf.reshape(x_d, [-1, self.hp.input_length * 2, self.hp.site_num, self.hp.emb_size])

        with tf.variable_scope('spatial_3'):
            spatial_3 = Transformer(self.hp)
            x_r = tf.reshape(x_r, [-1, self.hp.site_num, self.hp.features])
            x_r = spatial_3.encoder(inputs=x_r, input_length=self.hp.input_length)
            x_r = tf.reshape(x_r, [-1, self.hp.input_length, self.hp.site_num, self.hp.emb_size])

        x_long = tf.concat([x_w,x_d],axis=1)
        x_long = tf.transpose(x_long, [0, 2, 1, 3])
        x_long = tf.reshape(x_long,[-1, self.hp.input_length * 4, self.hp.emb_size])
        x_short = tf.transpose(x_r, [0, 2, 1, 3])
        x_short = tf.reshape(x_short, [-1, self.hp.input_length, self.hp.emb_size])

        self.e_lstm_1 = self.lstm()
        self.e_lstm_2 = self.lstm()

        with tf.variable_scope('encoder_lstm_1'):
            lstm_1_outpus, _ = tf.nn.dynamic_rnn(cell=self.e_lstm_1, inputs=x_long, dtype=tf.float32)
            x_long = lstm_1_outpus

        with tf.variable_scope('encoder_lstm_3'):
            lstm_3_outpus,_ = tf.nn.dynamic_rnn(cell=self.e_lstm_2, inputs=x_short, dtype=tf.float32)
            x_short = lstm_3_outpus

        print(x_long.shape)
        print(x_short.shape)

        A = self.attention(queries=x_short[:,-1:,:], keys=x_long, num_units=self.hp.emb_size)
        x = tf.concat([x_short[:,-1:,:], A], axis=-1)
        x = tf.reshape(x, [-1, self.hp.site_num, self.hp.emb_size*2])
        x = tf.layers.dense(x,units=self.hp.emb_size,activation=tf.nn.tanh,bias_initializer=tf.truncated_normal_initializer(dtype=tf.float32),name='hidden')
        return x

    def decoder(self, x):
        pre = tf.layers.dense(x,units=self.hp.output_length,name='output_y')
        return pre