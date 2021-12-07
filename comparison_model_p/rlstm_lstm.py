import comparison_model_p.rlstm as lstm
import tensorflow as tf
class rlstm_inf(object):
    def __init__(self,batch_size,predict_time=1, layer_num=1,nodes=128, is_training=True, placeholders=None):
        '''
        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:

        We need to define the encoder of Encoder-Decoder Model,and the parameter will be
        express in the Rlstm.
        '''
        self.batch_size=batch_size
        self.predict_time = predict_time
        self.layer_num=layer_num
        self.nodes=nodes
        self.is_training=is_training
        self.placeholders = placeholders
        self.decoder()

    def decoder(self):
        def cell():
            lstm_cell=tf.nn.rnn_cell.GRUCell(num_units=self.nodes)
            lstm_cell_=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1-self.placeholders['dropout'])
            return lstm_cell_
        self.d_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        self.d_initial_state = self.d_mlstm.zero_state(self.batch_size, tf.float32)

    def encoding(self, input):
        '''
        we always use c_state as the input to decoder
        '''
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            l=lstm.rlstm(batch_size=self.batch_size,layer_num=self.layer_num,nodes=self.nodes,is_training=self.is_training)
            (c_state,h_state)=l.calculate(input)
        return h_state

    def decoding(self,  encoder_hs):
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
            # to store the prediction results for road nodes on each time
            pres.append(results)

        return tf.concat(pres, axis=-1,name='output_y')