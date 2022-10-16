# -- coding: utf-8 --

import tensorflow as tf
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

class MultiLossLayer():
  def __init__(self, loss_list):
    self._loss_list = loss_list
    self._sigmas_sq = []
    for i in range(len(self._loss_list)):
      self._sigmas_sq.append(slim.variable('Sigma_sq_' + str(i), dtype=tf.float32, shape=[], initializer=tf.initializers.random_uniform(minval=0.2, maxval=1)))

  def get_loss(self):
    factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[0]))
    loss = tf.add(tf.multiply(factor, self._loss_list[0]), tf.log(self._sigmas_sq[0]))
    for i in range(1, len(self._sigmas_sq)):
      factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[i]))
      loss = tf.add(loss, tf.add(tf.multiply(factor, self._loss_list[i]), tf.log(self._sigmas_sq[i])))
    return loss

def get_loss(logits, ground_truths):
  multi_loss_class = None
  loss_list = []
  if FLAGS.use_multi_loss:
    loss_op, multi_loss_class = calc_multi_loss(loss_list)
  else:
    loss_op = loss_list[0]
    for i in range(1, len(loss_list)):
      loss_op = tf.add(loss_op, loss_list[i])
  return loss_op, loss_list, multi_loss_class

def calc_multi_loss(loss_list):
  multi_loss_layer = MultiLossLayer(loss_list)
  return multi_loss_layer.get_loss(), multi_loss_layer

def l1_masked_loss(logits, gt, mask):
  valus_diff = tf.abs(tf.subtract(logits, gt))
  L1_loss = tf.divide(tf.reduce_sum(valus_diff), tf.add(tf.reduce_sum(mask[:, :, :, 0]), 0.0001))
  return L1_loss

def loss(logits, labels, type='cross_entropy'):
  if type == 'cross_entropy':
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(cross_entropy, name='loss')
  if type == 'l2':
    return tf.nn.l2_loss(tf.subtract(logits, labels))
  if type == 'l1':
    return tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(logits, labels)), axis=-1))