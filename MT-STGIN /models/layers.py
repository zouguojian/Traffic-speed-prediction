# -- coding: utf-8 --
from models.inits import *
import tensorflow as tf


def sparse_dropout(x, keep_prob, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False, dim=64):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """

    if sparse:
        site_num = y.get_shape().as_list()[1]
        y = tf.transpose(y, perm=[1, 2, 0])
        y = tf.reshape(y, shape=[site_num, -1])
        res = tf.sparse_tensor_dense_matmul(x, y)
        y = tf.reshape(res, shape=[site_num, dim, -1])
        res = tf.transpose(y, perm=[2, 0, 1])
    else:
        shape = x.get_shape().as_list()  # [-1, site num, hidden size]
        x = tf.reshape(x, shape=[-1, shape[2]])
        res = tf.matmul(x, y)
        res = tf.reshape(res, shape=[-1, shape[1], y.get_shape().as_list()[1]])
    return res


class GraphConvolution():
    """
    Graph convolution layer.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 placeholders,
                 supports,
                 dropout=0.,
                 sparse_inputs=False,
                 act=tf.nn.relu,
                 bias=False,
                 featureless=False,
                 res_name='layer'):

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.vars = {}
        self.act = act
        self.support = supports
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.res_name = res_name

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        self.name = self.__class__.__name__.lower()
        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                # glorot() return: tf.Variable()
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def forward(self, inputs):
        x = inputs

        # dropout
        # if self.sparse_inputs:
        #     x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        # else:
        #     x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs, dim=self.input_dim)
            else:
                pre_sup = self.vars['weights_' + str(i)]

            support = dot(self.support[i], pre_sup, sparse=True, dim=self.output_dim)
            supports.append(support)

        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        # residual connection layer
        res_c = tf.layers.dense(inputs=inputs, units=self.output_dim, name=self.res_name)

        return tf.add(x=self.act(output), y=res_c)

        # return self.act(output)