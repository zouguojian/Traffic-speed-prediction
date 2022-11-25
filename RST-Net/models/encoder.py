# -- coding: utf-8 --
from models.st_block import ST_Block


class Encoder_ST(object):
    def __init__(self, hp, placeholders=None, model_func=None):
        '''
        :param hp:
        '''
        self.para = hp
        self.input_length = self.para.input_length
        self.placeholders = placeholders
        self.model_func = model_func

    def encoder_spatio_temporal(self, speed=None, STE=None, supports=None):
        '''
        :param features: [N, site_num, emb_size]
        :param STE: [N, input_length, site_num, emb_size]
        :return: [N, input_length, site_num, emb_size]
        '''
        # dynamic spatial correlation

        st_block = ST_Block(hp=self.para, placeholders=self.placeholders, input_length=self.input_length, model_func=self.model_func)

        if self.para.model_name == 'STGIN_2':
            result = st_block.spatio_temporal_(speed=speed,
                                               STE=STE,
                                               supports=supports)
        else:
            result = st_block.spatio_temporal(speed=speed,
                                              STE=STE,
                                              supports=supports)

        return result #[N, input_length, site_num, emb_size]