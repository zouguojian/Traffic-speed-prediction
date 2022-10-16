# -- coding: utf-8 --
from models.spatial_attention import SpatialTransformer
import tensorflow as tf
from models.temporal_attention import TemporalTransformer
from models.lstm import LstmClass

class ST_Block():
    def __init__(self, hp=None, placeholders=None, input_length=6, model_func=None):
        '''
        :param hp:
        '''
        self.para = hp
        self.batch_size = self.para.batch_size
        self.emb_size = self.para.emb_size
        self.site_num = self.para.site_num
        self.is_training = self.para.is_training
        self.dropout = self.para.dropout
        self.hidden_size = self.para.hidden_size
        self.hidden_layer =self.para.hidden_layer
        self.placeholders = placeholders
        self.input_length = input_length
        self.model_func = model_func

    def spatio_temporal_1(self, speed=None, pollution=None, day=None, hour=None, minute=None, position=None, supports=None):
        '''
        :param features: [N, site_num, emb_size]
        :param day: [N, input_length, site_num, emb_size]
        :param hour:
        :param position:
        :return: [N, input_length, site_num, emb_size]
        '''
        # dynamic spatial correlation
        x_s = speed
        S = SpatialTransformer(self.para)
        x_s = S.encoder(inputs=x_s,
                        input_length=self.input_length,
                        day=day,
                        hour=hour,
                        minute=minute,
                        position=position)

        # static spatial correlation
        x_g = tf.reshape(speed, shape=[-1, self.input_length, self.site_num, self.emb_size])
        x_g = tf.add_n([x_g, hour, minute, position])
        x_g = tf.reshape(x_g, shape=[-1, self.site_num, self.emb_size])
        gcn = self.model_func(self.placeholders,
                              input_dim=self.emb_size,
                              para=self.para,
                              supports=supports)
        x_g = gcn.predict(x_g)
        # x_g = tf.reshape(x_g, shape=[self.batch_size,
        #                              self.input_length,
        #                              self.site_num,
        #                              self.emb_size])
        print('encoder gcn outs shape is : ', x_g.shape)

        # feature fusion
        x_f = tf.concat([x_s, x_g], axis=-1)
        x_f = tf.layers.dense(x_f, units=self.emb_size, activation=tf.nn.tanh, name='fusion_1')

        """ --------------------------------------------------------------------------------------- """

        # this step use to encoding the input series data
        x_f = tf.reshape(x_f, shape=[self.batch_size, self.input_length, self.site_num, self.emb_size])
        x_p = tf.tile(input=tf.expand_dims(pollution, axis=2), multiples=[1, 1, self.site_num, 1])
        x = tf.concat([x_f, x_p], axis=-1)
        x = tf.layers.dense(x, units=self.emb_size, name='feed_1')
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, self.input_length, self.emb_size])

        # temporal correlation
        x_t = x
        T = TemporalTransformer(self.para)
        x_t = T.encoder(hiddens = x_t,
                        hidden = x_t,
                        hidden_units = self.emb_size,
                        dropout_rate = self.dropout,
                        is_training = self.is_training)

        # time series correlation
        x_l = x
        lstm_init = LstmClass(batch_size=self.batch_size * self.site_num,
                              layer_num=self.hidden_layer,
                              nodes=self.hidden_size,
                              placeholders=self.placeholders)
        x_l, c_states = lstm_init.encoding(x_l)

        # feature fusion
        x_f = tf.concat([x_t, x_l], axis=-1)
        x_f = tf.layers.dense(x_f, units=self.emb_size, activation=tf.nn.tanh, name='fusion_2')

        x_f = tf.reshape(x_f, shape=[self.batch_size, self.site_num, self.input_length, self.emb_size])
        x_f = tf.transpose(x_f, perm=[0, 2, 1, 3])
        return x_f #[N, input_length, site_num, emb_size]


    def spatio_temporal(self, speed=None, pollution=None, day=None, hour=None, minute=None, position=None, supports=None):
        '''
        :param features: [N, site_num, emb_size]
        :param day: [N, input_length, site_num, emb_size]
        :param hour:
        :param position:
        :return: [N, input_length, site_num, emb_size]
        '''
        # this step use to encoding the input series data
        speed = tf.reshape(speed, shape=[self.batch_size, self.input_length, self.site_num, self.emb_size])
        x_p = tf.tile(input=tf.expand_dims(pollution, axis=2), multiples=[1, 1, self.site_num, 1])
        x = tf.concat([speed, x_p], axis=-1)
        x = tf.layers.dense(x, units=self.emb_size, name='feed_1')
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, self.input_length, self.emb_size])

        # temporal correlation
        x_t = x
        T = TemporalTransformer(self.para)
        x_t = T.encoder(hiddens = x_t,
                        hidden = x_t,
                        hidden_units = self.emb_size,
                        num_heads=self.para.num_blocks,
                        num_blocks = self.para.num_heads,
                        dropout_rate = self.dropout,
                        is_training = self.is_training)

        # time series correlation
        x_l = x
        lstm_init = LstmClass(batch_size=self.batch_size * self.site_num,
                              layer_num=self.hidden_layer,
                              nodes=self.hidden_size,
                              placeholders=self.placeholders)
        x_l, c_states = lstm_init.encoding(x_l)

        # feature fusion
        x_f = tf.concat([x_t, x_l], axis=-1)
        x_f = tf.layers.dense(x_f, units=self.emb_size, activation=tf.nn.tanh, name='fusion_2')

        x_f = tf.reshape(x_f, shape=[self.batch_size, self.site_num, self.input_length, self.emb_size])
        x_f = tf.transpose(x_f, perm=[0, 2, 1, 3])

        """ --------------------------------------------------------------------------------------- """

        # dynamic spatial correlation
        x_f = tf.reshape(x_f, shape=[-1, self.site_num, self.emb_size])
        x_s = x_f
        S = SpatialTransformer(self.para)
        x_s = S.encoder(inputs=x_s,
                        input_length=self.input_length,
                        day=day,
                        hour=hour,
                        minute=minute,
                        position=position)
        # static spatial correlation
        x_g = tf.reshape(x_f, shape=[-1, self.input_length, self.site_num, self.emb_size])
        x_g = tf.add_n([x_g, hour, minute, position])
        x_g = tf.reshape(x_g, shape=[-1, self.site_num, self.emb_size])
        gcn = self.model_func(self.placeholders,
                              input_dim=self.emb_size,
                              para=self.para,
                              supports=supports)
        x_g = gcn.predict(x_g)
        # x_g = tf.reshape(x_g, shape=[self.batch_size,
        #                              self.input_length,
        #                              self.site_num,
        #                              self.emb_size])
        print('encoder gcn outs shape is : ', x_g.shape)

        # feature fusion
        x_f = tf.concat([x_s, x_g], axis=-1)
        x_f = tf.layers.dense(x_f, units=self.emb_size, activation=tf.nn.tanh, name='fusion_1')

        x_f = tf.reshape(x_f, shape=[self.batch_size, self.input_length, self.site_num, self.emb_size])

        return x_f #[N, input_length, site_num, emb_size]