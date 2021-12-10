import model.rlstm as lstm
import tensorflow as tf
class encoder(object):
    def __init__(self,input,batch_size,layer_num=1,nodes=128,is_training=True):
        '''
        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:

        We need to define the encoder of Encoder-Decoder Model,and the parameter will be
        express in the Rlstm.
        '''
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            encoder_Lstm=lstm.rlstm(layer_num,nodes,is_training)
            (self.c_state,self.h_state)=encoder_Lstm.calculate(input,batch_size)

    def encoding(self):
        '''
        we always use c_state as the input to decoder
        '''
        return (self.c_state,self.h_state)