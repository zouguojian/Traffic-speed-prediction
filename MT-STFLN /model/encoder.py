# -- coding: utf-8 --

import model.rlstm as lstm

class encoder(object):
    def __init__(self,
                 batch_size=32,
                 layer_num=1,
                 nodes=128,
                 is_training=True,
                 site_num=49):
        '''
        :param layer_num:
        :param nodes:
        :param is_training:

        We need to define the encoder of Encoder-Decoder Model,and the parameter will be
        express in the Rlstm.
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.is_training=is_training
        self.site_num=site_num

        self.encoder_lstm=lstm.rlstm(self.batch_size, self.layer_num, self.nodes, self.is_training,self.site_num)

    def encoding(self,inputs):
        '''
        we always use c_state as the input to decoder
        :param inputs:
        :param batch_size:
        :return:
        '''
        (self.c_states, self.h_states) = self.encoder_lstm.calculate(inputs)
        return (self.h_states, self.c_states)