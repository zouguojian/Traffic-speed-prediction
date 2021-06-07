# -- coding: utf-8 --
from gcn_model.inits import *
import tensorflow as tf


def sparse_dropout(x, keep_prob, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    print(dropout_mask.shape)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class GraphConvolution():
    """
    Graph convolution layer.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 placeholders,
                 dropout=0.,
                 sparse_inputs=False,
                 act=tf.nn.relu,
                 bias=False,
                 featureless=False):

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.vars = {}
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        self.name=self.__class__.__name__.lower()
        with tf.variable_scope(self.name+'_vars'):
            for i in range(len(self.support)):
                # glorot() return: tf.Variable()
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def forward(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]

            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)

        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)