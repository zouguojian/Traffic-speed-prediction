# -- coding: utf-8 --
from comparison_model.convlstm import ConvLSTMCell
import tensorflow as tf

class mul_convlstm(object):
    def __init__(self, batch, predict_time, shape=[49, 1], filters= 32 , kernel=[49, 1], layer_num=1, activation=tf.tanh, normalize=False, reuse=None):
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

        inputs=tf.expand_dims(inputs,axis=4)

        with tf.variable_scope(name_or_scope='encoder_convlstm',reuse=tf.AUTO_REUSE):
            # Add the ConvLSTM step.
            cell = ConvLSTMCell(self.shape, self.filters, self.kernel, normalize=self.normalize)

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

    def decoding(self, encoder_hs):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, site number, prediction size]
        '''

        h = []
        h_state = encoder_hs[:, -1, :, :, :]
        h_state = tf.expand_dims(input=h_state, axis=1)

        '''
        inputs shape is [batch size, 1, site num, features, out channel)
        '''
        # Add the ConvLSTM step.
        cell = ConvLSTMCell(self.shape, self.filters, self.kernel, normalize=self.normalize)
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

                results = tf.layers.dense(inputs=tf.squeeze(h_state), units=1, name='layer', reuse=tf.AUTO_REUSE)
            print(results.shape)
            h.append(results)
        pre=tf.concat(h,axis=-1)
        return pre

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
