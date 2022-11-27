# -- coding: utf-8 --
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from models.temporal_attention import TemporalTransformer

class InferenceClass(object):
    def __init__(self, para=None):
        self.para=para

    def weighs_add(self, inputs, hidden_size):
        # inputs size是[batch_size, max_time, encoder_size(hidden_size)]
        u_context = tf.Variable(tf.truncated_normal([hidden_size]), name='u_context')
        # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size]
        h = tf.layers.dense(inputs, hidden_size)
        # shape为[batch_size, max_time, 1]
        alpha = tf.nn.softmax(tf.reduce_sum(h, axis=2, keep_dims=True), dim=1)
        # alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
        # reduce_sum之前shape为[batch_size, max_time, hidden_size]，之后shape为[batch_size, hidden_size]
        atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
        return atten_output

    def inference(self, out_hiddens):
        '''
        :param out_hiddens: [N, output_length, site_num, emb_size]
        :return:
        '''
        
        if self.para.model_name == 'MT-STGIN':
            print(out_hiddens.shape)
            results_1 = tf.layers.dense(inputs=out_hiddens[:, 0:28], units=64, name='task_1', activation=tf.nn.relu)
            results_1 = tf.layers.dense(inputs=results_1, units=1, name='task_1_1')
            results_2 = tf.layers.dense(inputs=out_hiddens[:, 28: 52], units=64, name='task_2', activation=tf.nn.relu)
            results_2 = tf.layers.dense(inputs=results_2, units=1, name='task_2_1')
            results_3 = tf.layers.dense(inputs=out_hiddens[:, 52:], units=64, name='task_3', activation=tf.nn.relu)
            results_3 = tf.layers.dense(inputs=results_3, units=1, name='task_3_1')
            results_speed = tf.concat([results_1, results_2, results_3], axis=1)
            results_speed = tf.transpose(results_speed, [0, 2, 1, 3])
            results_speed = tf.squeeze(results_speed, axis=-1, name='output_y')
        else:
            results_speed = tf.layers.dense(inputs=tf.transpose(out_hiddens, [0, 2, 1, 3]), units=64, activation=tf.nn.relu, name='layer_spped_1')
            results_speed = tf.layers.dense(inputs=results_speed, units=1, activation=tf.nn.relu, name='layer_speed_2')
            results_speed = tf.squeeze(results_speed, axis=-1, name='output_y')

        return results_speed# [N, site_num, output_length]

    def dynamic_inference(self, features=None, STE=None):
        '''
        :param features: [N, output_length, site_num, emb_size]
        :return:
        '''
        pres = list()
        features = tf.reshape(tf.transpose(features, perm=[0, 2, 1, 3]),
                              shape=[-1, self.para.input_length, self.para.emb_size])
        for i in range(self.para.output_length):
            pre_features = STE[:,i:i+1,:,:]
            pre_features = tf.reshape(tf.transpose(pre_features, perm=[0, 2, 1, 3]),
                                      shape=[-1, 1, self.para.emb_size])  # 3-D

            print('in the decoder step, the input_features shape is : ', features.shape)
            print('in the decoder step, the pre_features shape is : ', pre_features.shape)
            T = TemporalTransformer(arg=self.para)
            x_t = T.encoder(hiddens = features,
                            hidden = pre_features) # [-1, 1, hidden_size]
            x = tf.squeeze(x_t, axis=1)
            x = tf.reshape(x, shape=[-1, self.para.site_num, self.para.emb_size])
            x = tf.layers.dense(inputs=x, units=64, name='layer_1', activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
            pre = tf.layers.dense(inputs=x, units=1, name='layer_2', activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
            pres.append(pre)

        return tf.concat(pres, axis=-1, name='output_y')