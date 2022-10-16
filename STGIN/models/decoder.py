# -- coding: utf-8 --
from models.st_block import ST_Block

class Decoder_ST(object):
    def __init__(self, hp, placeholders=None, model_func=None):
        '''
        :param hp:
        '''
        self.para = hp
        self.output_length = self.para.output_length
        self.placeholders = placeholders
        self.model_func = model_func

    def decoder_spatio_temporal(self, speed=None, STE=None, supports=None):
        '''
        :param features: [N, site_num, emb_size]
        :param day: [N, output_length, site_num, emb_size]
        :param hour:
        :param position:
        :return: [N, output_length, site_num, emb_size]
        '''
        # dynamic spatial correlation

        st_block = ST_Block(hp=self.para, placeholders=self.placeholders, input_length=self.output_length, model_func=self.model_func)
        result = st_block.spatio_temporal(speed=speed,
                                         STE = STE,
                                         supports=supports)
        return result #[N, output_length, site_num, emb_size]