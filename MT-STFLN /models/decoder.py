# -- coding: utf-8 --
from models.spatial_attention import SpatialTransformer
import tensorflow as tf
from models.temporal_attention import TemporalTransformer
from models.lstm import LstmClass


class Decoder_ST(object):
    def __init__(self, hp, placeholders=None, model_func=None):
        '''
        :param hp:
        '''
        self.para = hp
        self.batch_size = self.para.batch_size
        self.output_length = self.para.output_length
        self.emb_size = self.para.emb_size
        self.site_num = self.para.site_num
        self.hidden_size = self.para.hidden_size
        self.hidden_layer =self.para.hidden_layer
        self.placeholders = placeholders
        self.model_func = model_func

    def decoder_spatio_temporal(self, speed=None, pollution=None,day=None, hour=None, minute=None, position=None, supports=None):
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
                        input_length=self.output_length,
                        day=day,
                        hour=hour,
                        minute=minute,
                        position=position)

        # static spatial correlation
        x_g = tf.reshape(speed, shape=[-1, self.output_length, self.site_num, self.emb_size])
        x_g = tf.add_n([x_g, hour, minute, position])
        x_g = tf.reshape(x_g, shape=[-1, self.site_num, self.emb_size])
        gcn = self.model_func(self.placeholders,
                              input_dim=self.emb_size,
                              para=self.para,
                              supports=supports)
        x_g = gcn.predict(x_g)
        x_g = tf.reshape(x_g, shape=[self.batch_size,
                                     self.output_length,
                                     self.site_num,
                                     self.emb_size])
        print('encoder gcn outs shape is : ', x_g.shape)

        # feature fusion
        x_f = tf.concat([x_s, x_g], axis=-1)
        x_f = tf.layers.dense(x_f, units=self.emb_size, activation=tf.nn.tanh, name='fusion_1')

        """ --------------------------------------------------------------------------------------- """

        # this step use to encoding the input series data
        x_f = tf.reshape(x_f, shape=[self.batch_size, self.output_length, self.site_num, self.emb_size])
        x_p = tf.tile(input=tf.expand_dims(pollution, axis=2), multiples=[1, 1, self.site_num, 1])
        x = tf.concat([x_f, x_p], axis=-1)
        x = tf.layers.dense(x, units=self.emb_size, name='feed_1')
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, self.output_length, self.emb_size])

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

    def gcn_encoding(self, encoder_hs, site_num=None, x_p=None, day=None, hour=None, position=None):
        pres = list()
        pres_p=list()
        shape=encoder_hs.shape
        h_states=encoder_hs[:,-1,:,:]
        encoder_hs = tf.reshape(tf.transpose(encoder_hs, perm=[0, 2, 1, 3]),shape=[shape[0] * shape[2], shape[1], shape[3]])
        initial_state = self.initial_state

        for i in range(self.predict_time):
            # gcn for decoder processing, there is no question
            out_day=day[:,i,:,:]
            out_hour=hour[:,i,:,:]
            h_states = tf.layers.dense(inputs=h_states, units=out_day.shape[-1], reuse=tf.AUTO_REUSE)
            features=tf.add_n([h_states,out_day,out_hour,position[:,-1,:,:]])

            gcn_outs = gcn.predict(features) # gcn

            gan.input_length=1
            x = gan.encoder(speed=h_states, day=out_day, hour=out_hour, position=position[:,-1,:,:]) # gan

            features=tf.add_n([gcn_outs, x, position[:,-1,:,:]])
            features = tf.reshape(features, shape=[self.batch_size, 1, features.shape[-1]])

            print('features shape is : ',features.shape)

            h_state, state = tf.nn.dynamic_rnn(cell=self.mlstm_cell,inputs=features, initial_state=initial_state, dtype=tf.float32)
            initial_state = state

            # compute the attention state
            h_state = T_attention(hiddens=encoder_hs, hidden=h_state, hidden_units=shape[-1])  # attention # 注意修改
            # h_state = self.attention(h_t=h_state, encoder_hs=encoder_hs)  # attention
            h_states=tf.reshape(h_state,shape=[-1,site_num,self.nodes])

            pre_p=self.cnn(x_p)
            results = tf.layers.dense(inputs=h_state, units=1, name='layer', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            pre=tf.reshape(results,shape=[-1,site_num])
            # to store the prediction results for road nodes on each time
            pres.append(tf.expand_dims(pre, axis=-1))
            pres_p.append(pre_p)

        return tf.concat(pres, axis=-1,name='output_y'), tf.concat(pres_p,axis=-1,name='output_y_p')