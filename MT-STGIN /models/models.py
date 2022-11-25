# -- coding: utf-8 --

from models.layers import *
from models.metrics import *


class Model(object):
    def __init__(self):

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope('gcn'):  # add two dense layers "self.layers"
            self._build()

    def predict(self,inputs):
        '''
        :return:  output each node result :[batch size, n nodes, embedding]
        '''
        self.inputs = inputs  # input features
        # Build sequential layer model
        self.activations.append(self.inputs)

        for i, layer in enumerate(self.layers):
            hidden = layer.forward(self.activations[-1])
            # trick
            res_x=tf.layers.dense(self.inputs,units=hidden.shape[-1],name=str(i))
            self.activations.append(hidden+res_x) # feed forward
        outputs = self.activations[-1] # the last layer output
        return outputs

class GCN(Model):
    def __init__(self, placeholders, input_dim, para, supports=None):
        '''
        :param placeholders:
        :param input_dim:
        :param para:
        '''
        super(GCN, self).__init__()

        self.input_dim = input_dim  # input features dimension
        self.para=para

        # self.output_dim = placeholders['labels'].get_shape().as_list()[1]  # number of class
        # self.para.emb_size
        self.output_dim = input_dim # number of features of gcn output
        self.placeholders = placeholders
        self.supports=supports

        self.build()  # build model

    def _build(self):

        # self.layers.append(GraphConvolution(input_dim=self.input_dim,
        #                                     output_dim=self.para.hidden1,
        #                                     placeholders=self.placeholders,
        #                                     supports=self.supports,
        #                                     act=tf.nn.relu,
        #                                     bias=True,
        #                                     dropout=True,
        #                                     sparse_inputs=False,
        #                                     res_name='layer1'))
        #
        # self.layers.append(GraphConvolution(input_dim=self.input_dim,
        #                                     output_dim=self.output_dim,
        #                                     placeholders=self.placeholders,
        #                                     supports=self.supports,
        #                                     act=tf.nn.relu,
        #                                     bias=True,
        #                                     dropout=True,
        #                                     sparse_inputs=False,
        #                                     res_name='layer3'))
        #
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            supports=self.supports,
                                            act=tf.nn.relu,
                                            bias=True,
                                            dropout=True,
                                            sparse_inputs=False,
                                            res_name='layer4'))

        self.layers.append(GraphConvolution(input_dim=self.output_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            supports=self.supports,
                                            act=lambda x: x,
                                            dropout=False,
                                            res_name='layer5'))