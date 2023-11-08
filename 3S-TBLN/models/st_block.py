# -- coding: utf-8 --
from models.utils import *
from models.inits import *


def channel_combine(x=None):
    '''
    :param x: [-1, channel, site, dim]
    :return: [-1, 1, site, dim * channel number]
    '''
    x = tf.concat(tf.split(x, channel, axis=1), axis=-1)
    return x


def st_attention(x, is_encoder=True, pre_x=None, channels=3, input_len=12):
    '''
    :param x:  [-1, len, site, dim]
    :return: [-1, len, site, dim * channel number]
    '''
    global channel
    channel = channels
    if channels > 1:
        x = tf.concat([x[:, -channels:], x], axis=1)
    x = tf.concat(list(map(channel_combine, [x[:, i:i + channels] for i in range(input_len)])), axis=1)

    return x


def siteCombine(x=None, channels=3):
    '''
    :param x: [-1, channel, site, dim]
    :return: [-1, 1, site * channel number, dim]
    '''
    x = tf.concat(tf.split(x, channels, axis=1), axis=2)
    return x


def STHolistic(x, is_encoder=True, pre_x=None, channels=3, input_len=12):
    '''
    :param x: [-1, len, site, dim]
    :return: [-1, len, site * channel number, dim]
    '''
    if channels > 1 and is_encoder:
        x = tf.concat([x[:, -channels:], x], axis=1)
    elif channels > 1:
        x = tf.concat([pre_x, x], axis=1)
    x = tf.concat([siteCombine(x[:, i:i + channels], channels=channels) for i in range(input_len)], axis=1)
    return x


def fusionGate(x, y):
    '''
    :param x: [-1, len, site, dim]
    :param y: [-1, len, site, dim]
    :return: [-1, len, site, dim]
    '''
    z = tf.nn.sigmoid(tf.multiply(x, y))
    h = tf.add(tf.multiply(z, x), tf.multiply(1 - z, y))
    return h


def FC(x, units, activations, bn, bn_decay, is_training, use_bias=True, drop=None):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        if drop is not None:
            x = dropout(x, drop=drop, is_training=is_training)
        x = conv2d(
            x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn=bn, bn_decay=bn_decay, is_training=is_training)
    return x


def STEmbedding(SE, TE, T, D, bn, bn_decay, is_training):
    '''
    spatio-temporal embedding
    SE:     [N, D]
    TE:     [batch_size, P + Q, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, P + Q, N, D]
    '''
    # spatial embedding
    # SE = tf.expand_dims(tf.expand_dims(SE, axis=0), axis=0)
    SE = FC(
        SE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # temporal embedding
    TE = tf.concat(TE, axis=-1)
    TE = FC(
        TE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return tf.add(SE, TE)


def spatialAttention(X, Key, K, d, bn, bn_decay, is_training, top_k=32):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        Key, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        Key, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # [K * batch_size, num_step, N, N]
    attention = tf.matmul(query, key, transpose_b=True)
    attention /= (d ** 0.5)

    values, indices = tf.math.top_k(input=attention, k=top_k)
    min_ = tf.reduce_min(values, axis=-1, keepdims=True)
    attention = tf.where(tf.math.greater_equal(attention, min_), attention, tf.ones_like(attention) * (-2 ** 32 + 1))

    # a_top, a_top_idx = tf.nn.top_k(input=attention, k=top_k, sorted=False)
    # kth = tf.reduce_min(a_top, axis=-1, keepdims=True)
    # top2 = tf.greater_equal(attention, kth)
    # mask = tf.cast(top2, dtype=tf.float32)
    # # 只保留张量top k的值不变，其余值变为0
    # attention = tf.multiply(attention, mask)

    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def temporalAttention(X, K, d, bn, bn_decay, is_training, mask=True):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # query: [K * batch_size, N, num_step, d]
    # key:   [K * batch_size, N, d, num_step]
    # value: [K * batch_size, N, num_step, d]
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    # [K * batch_size, N, num_step, num_step]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    # mask attention score
    if mask:
        batch_size = tf.shape(X)[0]
        num_step = X.get_shape()[1].value
        N = X.get_shape()[2].value
        mask = tf.ones(shape=(num_step, num_step))
        mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
        mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=0)
        mask = tf.tile(mask, multiples=(K * batch_size, N, 1, 1))
        mask = tf.cast(mask, dtype=tf.bool)
        attention = tf.compat.v2.where(
            condition=mask, x=attention, y=-2 ** 15 + 1)
    # softmax
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    '''
    gated fusion
    HS:     [batch_size, num_step, N, D]
    HT:     [batch_size, num_step, N, D]
    D:      output dims
    return: [batch_size, num_step, N, D]
    '''
    XS = FC(
        HS, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=False)
    XT = FC(
        HT, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=True)
    z = tf.nn.sigmoid(tf.add(XS, XT))
    H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
    H = FC(
        H, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return H


def BridgeTrans(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training):
    '''
    transform attention mechanism
    X:      [batch_size, P, N, D]
    STE_P:  [batch_size, P, N, D]
    STE_Q:  [batch_size, Q, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, Q, N, D]
    '''
    D = K * d
    # query: [batch_size, Q, N, K * d]
    # key:   [batch_size, P, N, K * d]
    # value: [batch_size, P, N, K * d]
    query = FC(
        STE_Q, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        STE_P, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # query: [K * batch_size, Q, N, d]
    # key:   [K * batch_size, P, N, d]
    # value: [K * batch_size, P, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # query: [K * batch_size, N, Q, d]
    # key:   [K * batch_size, N, d, P]
    # value: [K * batch_size, N, P, d]
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    # [K * batch_size, N, Q, P]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, Q, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def STAttBlock(X, STE, K, d, bn, bn_decay, is_training, mask=True, top_k=32, is_encoder=True, pre_x=None, N=207,
               channels=3, input_len=12):
    # XH = tf.concat((X, STE), axis=-1)
    XH = X
    HT = temporalAttention(XH, K, d, bn, bn_decay, is_training, mask=mask)
    # 将一段时间内的不同时间不同站点的特征进行拼接，再用全局时空注意力计算, idea 1
    XS = tf.concat((X, STE), axis=-1)
    XS = STHolistic(XS, is_encoder=is_encoder, pre_x=pre_x, channels=channels, input_len=input_len)
    print(XS[:, :, -N:].shape, XS.shape)
    HS = spatialAttention(XS[:, :, -N:], XS, K, d, bn, bn_decay, is_training, top_k=top_k)
    H = fusionGate(HS, HT)
    # H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training)
    return tf.add(X, H)


def TS_TBLN(XS, XS_All, TE, SE, P, Q, T, L, K, d, bn, bn_decay, is_training, top_k=32, N=207, channels=3):
    '''
    3S-TBLN
    X：       [batch_size, P, N]
    TE：      [batch_size, P + Q, 2] (time-of-day, day-of-week)
    SE：      [N, K * d]
    P：       number of history steps
    Q：       number of prediction steps
    T：       one day is divided into T steps
    L：       number of STAtt blocks in the encoder/decoder
    K：       number of attention heads
    d：       dimension of each attention head outputs
    return：  [batch_size, Q, N]
    '''
    D = K * d
    # input
    X = FC(
        XS, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)  # [-1, P, N, dim]
    XS = X
    X_All = FC(
        XS_All, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)  # [-1, P + Q, N, dim]
    # STE
    STE = STEmbedding(SE, TE, T, D, bn, bn_decay, is_training)
    STE_P = STE[:, :P]
    STE_Q = STE[:, P:]

    # encoder
    for i in range(L):
        with tf.variable_scope("encoder_num_blocks_{}".format(i)):
            X = STAttBlock(X + X_All[:, :P], STE_P, K, d, bn, bn_decay, is_training, top_k=top_k, N=N,
                           channels=channels, input_len=P)
    print('encoder output shape is ', X.shape)

    # BridgeTrans encoder
    with tf.variable_scope("BridgeTrans_Encoder_1"):
        X = BridgeTrans(X, X + STE_P, STE_Q + X_All[:, P:], K, d, bn, bn_decay, is_training)
    print('bridge output shape is ', X.shape)

    # decoder
    for i in range(L):
        with tf.variable_scope("decoder_num_blocks_{}".format(i)):
            X = STAttBlock(X + X_All[:, P:], STE_Q, K, d, bn, bn_decay, is_training,
                           top_k=top_k, is_encoder=False,
                           pre_x=tf.concat([XS[:, -channels:] + X_All[:, P - channels:P], STE_P[:, -channels:]],
                                           axis=-1),
                           N=N, channels=channels, input_len=Q)
    print('decoder output shape is ', X.shape)
    X_en = X

    # BridgeTrans decoder
    with tf.variable_scope("BridgeTrans_Decoder_1"):
        X = BridgeTrans(X, X + STE_Q, STE_P + X_All[:, :P], K, d, bn, bn_decay, is_training)
    print('decoder bridge output shape is ', X.shape)
    X_de = X

    # inference
    X_en = FC(
        X_en, units=[D, 1], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training,
        use_bias=True, drop=0.1)

    X_de = FC(
        X_de, units=[D, 1], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training,
        use_bias=True, drop=0.1)
    X = tf.concat([X_de, X_en], axis=1)
    return tf.squeeze(X, axis=3)