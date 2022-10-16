# -- coding: utf-8 --
from baseline.mdl.convlstm import ConvLSTMCell
import tensorflow as tf

class mul_convlstm(object):
    def __init__(self, batch, predict_time, shape=[108, 5], filters= 32 , kernel=[108, 2], layer_num=1, activation=tf.tanh, normalize=True, reuse=None):
        self.batch=batch
        self.predict_time=predict_time
        self.layers=layer_num
        self.activation=activation
        self.normalize=normalize
        self.reuse=reuse

        self.shape = shape
        self.kernel = kernel
        self.filters = filters        # numbers of output channel

    def encoding(self, inputs):
        '''
        :return: shape is [batch size, time size, site num, features, out channel)
        '''

        # inputs=tf.expand_dims(inputs,axis=4)

        with tf.variable_scope(name_or_scope='encoder_convlstm',reuse=tf.AUTO_REUSE):
            # Add the ConvLSTM step.
            cell = ConvLSTMCell(self.shape, self.filters, self.kernel)

            '''
            inputs shape is : [batch size, time size, site number, features, input channel]
            outputs is : [batch size, time size, site number, features, output channel]
            state: LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(32, 162, 5, 12) dtype=float32>, 
                    h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(32, 162, 5, 12) dtype=float32>)
            '''
            init_state=cell.zero_state(self.batch,tf.float32)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state, dtype=inputs.dtype)

            print(outputs.shape)
            print(state)
        return outputs

    def cnn(self, x=None):
        '''
        :param x: [batch, output_length, site_num, features, emb_size]
        :return:
        '''
        x = tf.squeeze(x, axis=3)
        with tf.variable_scope('cnn_layer', reuse=tf.AUTO_REUSE):
            layer1 = tf.layers.conv2d(inputs=x,
                                         filters=self.filters,
                                         kernel_size=[3,3],
                                         padding='same',
                                         kernel_initializer=tf.initializers.truncated_normal(),
                                         activation=tf.nn.relu,
                                         name='conv_1')
            # filter1 = tf.Variable(initial_value=tf.random_normal(shape=[3, 108, self.para.emb_size, 64]), name='fitter_1')
            # layer1 = tf.nn.conv2d(input=x, filter=filter1, strides=[1, 1, 1, 1], padding='SAME')
            # layer1 = tf.nn.sigmoid(layer1)
            print('layer1 shape is : ', layer1.shape)

            layer2 = tf.layers.conv2d(inputs=layer1,
                                         filters=self.filters,
                                         kernel_size=[3,3],
                                         padding='same',
                                         kernel_initializer=tf.initializers.truncated_normal(),
                                         activation=tf.nn.relu,
                                         name='conv_2')
            print('layer2 shape is : ', layer2.shape)

            full_1 = tf.layers.dense(inputs=layer2, units=64, name='layer_1', activation=tf.nn.relu)
            results = tf.layers.dense(inputs=full_1, units=1, name='layer_2', activation=tf.nn.relu)

            pres = tf.squeeze(results,axis=-1, name='output_y')
        return tf.transpose(pres, perm=[0, 2, 1])

    def decoding(self, encoder_hs):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, site number, prediction size]
        '''

        h = []
        outpus = list()
        h_state = encoder_hs[:, -1, :, :, :]
        h_state = tf.expand_dims(input=h_state, axis=1)
        with tf.variable_scope(name_or_scope='decoder_convlstm',reuse=tf.AUTO_REUSE):
            '''
            inputs shape is [batch size, 1, site num, features, out channel)
            '''
            # Add the ConvLSTM step.
            cell = ConvLSTMCell(self.shape, self.filters, self.kernel)
            init_state = cell.zero_state(self.batch, tf.float32)

            '''
            inputs shape is : [batch size, 1, site number, features, input channel]
            outputs is : [batch size, 1, site number, features, output channel]
            state: LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(32, 162, 5, 12) dtype=float32>, 
                    h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(32, 162, 5, 12) dtype=float32>)
            '''
            for i in range(self.predict_time):
                ''' return shape is [batch size, time size, height, site num, out channel) '''
                with tf.variable_scope('decoder_lstm', reuse=tf.AUTO_REUSE):
                    h_state, state = tf.nn.dynamic_rnn(cell=cell, inputs=h_state,
                                                        initial_state=init_state, dtype=tf.float32)
                init_state = state
                outpus.append(h_state)

            pres = self.cnn(x=tf.concat(outpus, axis=1))
            return pres

if __name__ == '__main__':

    batch_size = 32
    timesteps = 3
    shape = [162, 5]
    kernel = [162, 2]
    channels = 1
    filters = 12        # numbers of output channel

    # Create a placeholder for videos.
    inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape)

    multi=mul_convlstm(batch=32, predict_time=2)

    hs=multi.encoding(inputs)
    print(hs.shape)
    pre =multi.decoding(hs)
    print(pre.shape)
