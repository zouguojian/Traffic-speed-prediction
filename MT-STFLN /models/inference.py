# -- coding: utf-8 --
import tensorflow as tf

class InferenceClass(object):
    def __init__(self, para=None):
        self.para=para

    def cnn(self,x=None):
        '''
        :param x: [N, output_length, site_num, emb_size]
        :return:
        '''
        with tf.variable_scope('cnn_layer', reuse=tf.AUTO_REUSE):
            filter1 = tf.Variable(initial_value=tf.random_normal(shape=[6, 108, self.para.emb_size, 64]), name='fitter_1')
            layer1 = tf.nn.conv2d(input=x, filter=filter1, strides=[1, 1, 1, 1], padding='SAME')
            layer1 = tf.nn.relu(layer1)
            print('layer1 shape is : ', layer1.shape)

            filter2 = tf.Variable(initial_value=tf.random_normal(shape=[6, 108, 64, 32]), name='fitter_1')
            layer2 = tf.nn.conv2d(input=layer1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME')
            layer2 = tf.nn.relu(layer2)
            print('layer2 shape is : ', layer2.shape)

            filter3 = tf.Variable(initial_value=tf.random_normal(shape=[6, 108, 32, 8]), name='fitter_1')
            layer3 = tf.nn.conv2d(input=layer2, filter=filter3, strides=[1, 1, 1, 1], padding='SAME')
            layer3 = tf.nn.relu(layer3)
            print('layer3 shape is : ', layer3.shape)

            cnn_shape = layer3.get_shape().as_list()
            nodes = cnn_shape[2] * cnn_shape[3]
            cnn_out = tf.reshape(layer3, [-1, cnn_shape[1], nodes])

            results_pollution=tf.layers.dense(inputs=cnn_out, units=128, name='layer_pollution_1')
            results_pollution=tf.layers.dense(inputs=results_pollution, units=1, name='layer_pollution_2')
            results_pollution=tf.squeeze(results_pollution,axis=-1)
        return results_pollution

    def inference(self, out_hiddens):
        '''
        :param out_hiddens: [N, output_length, site_num, emb_size]
        :return:
        '''
        results_speed = tf.layers.dense(inputs=tf.transpose(out_hiddens, [0, 2, 1, 3]), units=128, name='layer_spped_1')
        results_speed = tf.layers.dense(inputs=results_speed, units=1, name='layer_speed_2')
        results_speed = tf.squeeze(results_speed, axis=-1, name='output_y')

        results_pollution = self.cnn(x=out_hiddens)

        return results_speed, results_pollution # [N, site_num, output_length], [N, output_length]