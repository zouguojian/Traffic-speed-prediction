# -- coding: utf-8 --
import tensorflow as tf
def gate_attention(self, inputs, hidden_size):
    # inputs size是[batch_size, max_time, encoder_size(hidden_size)]
    u_context = tf.Variable(tf.truncated_normal([hidden_size]), name='u_context')
    # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size]
    h = tf.layers.dense(inputs, hidden_size, activation_fn=tf.nn.tanh)
    # shape为[batch_size, max_time, 1]
    alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
    # reduce_sum之前shape为[batch_size, max_time, hidden_size]，之后shape为[batch_size, hidden_size]
    atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
    return atten_output