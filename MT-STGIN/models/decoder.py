# -- coding: utf-8 --
from models.st_block import ST_Block
from models.inits import *
from models.bridge import BridgeTransformer

class Decoder_ST(object):
    def __init__(self, hp, placeholders=None, model_func=None):
        '''
        :param hp:
        '''
        self.para = hp
        self.output_length = self.para.output_length
        self.placeholders = placeholders
        self.model_func = model_func

    def decoder_spatio_temporal(self, speed=None, STE=None, supports=None,causality=False):
        '''
        :param speed: [N, time length, site_num, emb_size]
        :param day: [N, output_length, site_num, emb_size]
        :param hour:
        :param position:
        :return: [N, output_length, site_num, emb_size]
        '''
        # dynamic spatial correlation
        if self.para.model_name != 'MT-STGIN-4':
            st_block = ST_Block(hp=self.para, placeholders=self.placeholders, input_length=self.output_length + self.para.pre_length,
                            model_func=self.model_func)
            result = st_block.spatio_temporal(speed=speed,
                                             STE = STE,
                                             supports=supports,causality=causality)
        else:
            bridge_encoder = BridgeTransformer(self.para)
            st_block = ST_Block(hp=self.para, placeholders=self.placeholders, input_length=self.output_length,
                            model_func=self.model_func)
            result=list()
            """注意此处，需要将pre_length设置为0才行"""
            for time_step in range(self.output_length):
                with tf.variable_scope('dynamic_decoding_bridge',reuse=tf.AUTO_REUSE):
                    bridge_outs = bridge_encoder.encoder(X=speed,
                                                         X_P=speed,
                                                         X_Q=STE[:,time_step:time_step+1],
                                                         causality=False)
                    # print('prediction time step is : ',time_step)
                    # print(bridge_outs.shape)

                with tf.variable_scope('dynamic_decoding_spatio_temporal', reuse=tf.AUTO_REUSE):
                    each_time_step_hidden = st_block.dynamic_spatio_temporal(speed=bridge_outs,
                                                                             STE = STE[:,time_step:time_step+1],
                                                                             supports=supports,causality=causality)
                    # print(each_time_step_hidden.shape)
                speed = tf.concat([speed, each_time_step_hidden], axis=1)
                result.append(each_time_step_hidden)
            result =tf.concat(result, axis=1)

        return result #[N, output_length, site_num, emb_size]