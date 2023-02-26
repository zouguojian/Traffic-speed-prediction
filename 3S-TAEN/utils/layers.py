# -- coding: utf-8 --
'''
tf.layers.conv1d(
inputs,
filters,
kernel_size,
strides=1,
)
inputs： 输入tensor， 维度(batch_size, seq_length, embedding_dim) 是一个三维的tensor；
其中，batch_size指每次输入的文本数量；seq_length指每个文本的词语数或者单字数；
embedding_dim指每个词语或者每个字的向量长度；
例如每次训练输入2篇文本，每篇文本有100个词，每个词的向量长度为20，那input维度即为(2, 100, 20)。
filters：过滤器（卷积核）的数目
kernel_size：卷积核的大小，卷积核本身应该是二维的，这里只需要指定一维，因为第二个维度即长度与词向量的长度一致，
卷积核只能从上往下走，不能从左往右走，即只能按照文本中词的顺序，也是列的顺序。
'''
import tensorflow as tf

conv1d = tf.layers.conv1d


def attn_head(seq=None, out_sz=None, bias_mat=None, activation=tf.nn.elu, in_drop=0.0, coef_drop=0.0, residual=False):
    '''
    self attention
    :param seq: shape is [batch_size, seq_length, embedding_dim]
    :param out_sz:
    :param bias_mat:
    :param activation:
    :param in_drop:   dropout
    :param coef_drop: dropout
    :param residual:
    :return:
    '''

    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])

        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation


if __name__=='__main__':
    x=tf.placeholder(dtype=tf.float32,shape=[32,16,300])
    att=attn_head(seq=x,out_sz=[10,20],)
