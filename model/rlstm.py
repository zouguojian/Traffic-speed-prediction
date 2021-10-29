# -- coding: utf-8 --
import tensorflow as tf

class rlstm(object):
    def __init__(self,batch_size, layer_num=1,nodes=128,is_training=None,site_num=49):
        self.batch_size=batch_size
        self.layer_num=layer_num    #the numbers of layer
        self.nodes=nodes            #the numbers of nodes to each layer
        self.is_training=is_training
        self.site_num=49
        if self.is_training: self.train_state(self.batch_size)
        self.test_state(self.site_num)
        print(self.is_training)

    #the funtion of init_state() is used to initialized the state of c and h
    def train_state(self, batch_size=32):
        '''
        :return:
        '''
        with tf.variable_scope(name_or_scope='train_state', reuse=tf.AUTO_REUSE):
            self.c_state_train = tf.Variable(tf.zeros(shape=[batch_size,self.nodes],dtype=tf.float32),name='c_state')
            self.h_state_train = tf.Variable(tf.zeros(shape=[batch_size,self.nodes],dtype=tf.float32),name='h_state')

        # return c_state_train,h_state_train

    def test_state(self, batch_size=1):
        '''
        :return:
        '''
        with tf.variable_scope(name_or_scope='test_state', reuse=tf.AUTO_REUSE):
            self.c_state_test = tf.Variable(tf.zeros(shape=[batch_size,self.nodes],dtype=tf.float32),name='c_state')
            self.h_state_test = tf.Variable(tf.zeros(shape=[batch_size, self.nodes], dtype=tf.float32), name='h_state')

        # return c_state_test, h_state_test

    def lstm_layer(self,inputs,c_state, h_state, layer):
        '''

        :param input:
        :return:
        '''
        input = tf.concat([inputs, c_state, h_state], axis=1)
        #the fist gate, read gate
        #output shape is [batch_size,h+x+c]

        read_gate=tf.layers.dense(inputs=input,
                                  units=input.shape[1],
                                  activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  bias_initializer=tf.constant_initializer(0.0),
                                  name='read_gate'+str(layer),reuse=tf.AUTO_REUSE)
        '''
        #contral data is that the read gate to control data stream,
        #read the important information in the inputs.
        '''
        control_data=tf.multiply(read_gate,input)

        '''
        #forget gate, it is means that use the read gate output 
        and the model inputs to forget the un useless data .
        input size is[batch_size,h+x+c], the output size is [batch_size,nodes]
        '''

        forget_gate=tf.layers.dense(inputs=control_data,
                                    units=self.nodes,
                                    activation=tf.nn.sigmoid,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    name='forget_gate'+str(layer),
                                    reuse=tf.AUTO_REUSE)

        # write opration is used to update the cell state
        # the output size is :[batch_size,nodes]
        write=tf.layers.dense(inputs=control_data,
                              units=self.nodes,
                              activation=tf.nn.sigmoid,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                              bias_initializer=tf.constant_initializer(0.0),
                              name='write'+str(layer),
                              reuse=tf.AUTO_REUSE)
        '''
        it used to update the cell state,c
        and the shape is [batch_size,nodes]
        '''
        c_hat=tf.layers.dense(inputs=control_data,
                              units=self.nodes,
                              activation=tf.nn.tanh,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                              bias_initializer=tf.constant_initializer(0.0),
                              name='c_hate'+str(layer),
                              reuse=tf.AUTO_REUSE)

        '''
        the final cell state c, and used to reference as the O(output)
        the size is [batch_size,nodes] 
        '''

        self.c=tf.multiply(forget_gate,c_state)+tf.multiply(write,c_hat)

        '''
        We can used this formula to achieve the traditional lSTM output, it combine c_hat and control_data
        the size of out is[batch_size,nodes]
        '''
        # O=tf.layers.dense(inputs=control_data,units=self.nodes,
        #                           activation=tf.nn.tanh,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                           bias_initializer=tf.constant_initializer(0.0),name='out')
        # self.H=tf.multiply(O,tf.tanh(self.c))
        #
        self.h=tf.tanh(self.c)

        return (self.h, self.c)

    def calculate(self,input):
        '''
        this function
        :param input: input,and the size is [bacth_size,time_size,data_features]
        :return:
        '''

        self.c_states=[]
        self.h_states=[]

        '''
        use two layer loop ,the first loop to divided the layer,and the second layer loop 
        used to extract the time series features
        '''
        for layer in range(self.layer_num):
            with tf.variable_scope(name_or_scope=str(layer)):
                if not self.is_training: c_state,h_state=self.c_state_test,self.h_state_test
                else:c_state,h_state=self.c_state_train,self.h_state_train

                h = []
                c = []
                for time in range(input.shape[1]):
                    (c_state,h_state)=self.lstm_layer(input[:,time,:],c_state,h_state, layer)
                    h.append(h_state)
                    c.append(c_state)
                input=tf.transpose(tf.convert_to_tensor(h,dtype=tf.float32),[1,0,2])

                #the state of each layer, the end time
                self.c_states.append(c_state)
                self.h_states.append(h_state)
            if layer==self.layer_num-1:return (c,h)
        return (self.c_states,self.h_states)


# x=tf.Variable(tf.constant([[[2,3,4,5],[2,3,4,5]]],dtype=tf.float32))
# print(x.shape)
# lstm=rlstm(1,2,256)
# result=lstm.calculate(x)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(result).shape)