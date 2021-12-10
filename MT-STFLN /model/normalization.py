# -- coding: utf-8 --
import tensorflow as tf

class Normalization(object):
    def __init__(self,inputs,out_size,is_training=False):
        self.inputs=inputs
        self.out_size=out_size
        self.is_training=is_training

    def feed_forward(self):
        '''
        the feed forward layer
        :return: [batch,time_size,filed_size,new_features]
        '''
        inputs_shape = self.inputs.get_shape().as_list()  # list type
        with tf.variable_scope('w',reuse=tf.AUTO_REUSE):
            w=tf.Variable(tf.random_normal(shape=[inputs_shape[-1], self.out_size], stddev=0.1, dtype=tf.float32))
        self.inputs=tf.reshape(self.inputs,shape=[-1,inputs_shape[-1]]) #改变数据维度，为了后序的tf.matmul(self.inputs,w)的计算
        self.inputs=tf.nn.tanh(tf.matmul(self.inputs,w))  #这里可以添加激活函数，如：tanh
        inputs_shape[-1]=self.out_size
        self.inputs=tf.reshape(self.inputs,shape=[i if i!=None else -1 for i in inputs_shape]) #变成为原来输出时候的维度
        return self.inputs

    def normal(self):
        '''
        if is_training is true ,use the normalization function
        else: do not
        :return:
        '''
        if self.is_training:
            # self.inputs=tf.sparse_to_dense(self.inputs,)
            self.inputs=tf.layers.batch_normalization(self.inputs, training=True)