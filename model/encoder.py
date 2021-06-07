# -- coding: utf-8 --

import model.rlstm as lstm

class encoder(object):
    def __init__(self,
                 layer_num=1,
                 nodes=128,
                 is_training=True):
        '''
        :param layer_num:
        :param nodes:
        :param is_training:

        We need to define the encoder of Encoder-Decoder Model,and the parameter will be
        express in the Rlstm.
        '''

        self.encoder_lstm=lstm.rlstm(layer_num,nodes,is_training)

    def encoding(self,inputs, batch_size):
        '''
        we always use c_state as the input to decoder
        :param inputs:
        :param batch_size:
        :return:
        '''
        (self.c_states, self.h_states) = self.encoder_lstm.calculate(inputs, batch_size)
        return (self.c_states,self.h_states)