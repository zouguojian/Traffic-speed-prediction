import tensorflow as tf
class lstm(object):
    def __init__(self,input,batch_size,layer_num=1,nodes=128,is_training=True):
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
        self.input=input

    def keep_pro(self):
        '''
        used to define the self.keepProb value
        :return:
        '''
        if self.is_training:self.keepProb=0.5
        else:self.keepProb=1.0

    def lstm_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)
        # if confirg.KeepProb<1:
        return tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keepProb)

    def encoding(self):
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        self.initial_state=mlstm_cell.zero_state(self.batch_size,tf.float32)

        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            ouputs, state = tf.nn.dynamic_rnn(cell=mlstm_cell, inputs=self.input, initial_state=self.initial_state,dtype=tf.float32)
        # we use the list h to recoder the out of decoder eatch time
        return state[-1][0]