# -- coding: utf-8 --
from models.inits import tf

class SemanticTransformer(object):
    def __init__(self, input_len=12, emb_size=128, site_num=66, features=1):
        self.input_len=input_len
        self.emb_size=emb_size
        self.site_num =site_num
        self.features =features

    def transfer(self, speed=None):
        speed = tf.reshape(speed, [-1, self.input_len, self.features]) # [-1, input_length, features]
        speed1 = tf.layers.conv1d(inputs=speed,
                                  filters=self.emb_size,
                                  kernel_size=2,
                                  padding='SAME',
                                  kernel_initializer=tf.truncated_normal_initializer(),
                                  name='conv_1', )
        speed2 = tf.layers.conv1d(inputs=speed,
                                  filters=self.emb_size,
                                  kernel_size=3,
                                  padding='SAME',
                                  kernel_initializer=tf.truncated_normal_initializer(),
                                  name='conv_2')
        speed3 = tf.layers.conv1d(inputs=speed,
                                  filters=self.emb_size,
                                  kernel_size=1,
                                  padding='SAME',
                                  kernel_initializer=tf.truncated_normal_initializer(),
                                  name='conv_3')
        speed = tf.add_n([speed1, speed2, speed3])
        speed = tf.nn.relu(speed)
        speed = tf.reshape(speed, [-1, self.site_num, self.input_len, self.emb_size])
        speed = tf.transpose(speed, perm=[0, 2, 1, 3]) #[-1, input_len, site num, emb_size]
        return speed