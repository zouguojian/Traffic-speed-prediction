# -- coding: utf-8 --
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from tf_utils import *
import seaborn as sns

def construct_feed_dict(xs, xs_all, label_s, d_of_week, day, hour, minute, mask=[],placeholders=None, site =207, is_training=True):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['position']: np.array([[i for i in range(site)]],dtype=np.int32)})
    feed_dict.update({placeholders['labels']: label_s})
    feed_dict.update({placeholders['week']: d_of_week})
    feed_dict.update({placeholders['day']: day})
    feed_dict.update({placeholders['hour']: hour})
    feed_dict.update({placeholders['minute']: minute})
    feed_dict.update({placeholders['features']: xs})
    feed_dict.update({placeholders['features_all']: xs_all})
    feed_dict.update({placeholders['random_mask']: mask})
    feed_dict.update({placeholders['is_training']: is_training})
    return feed_dict

import matplotlib.pyplot as plt

def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)

        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label.astype(np.float32))
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (pred - np.mean(pred)))) / (np.std(pred) * np.std(label))
        sse = np.sum((label - pred) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        r2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
    return mae, rmse, mape

import numpy as np
import pandas as pd

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def loadData(args):
    # Traffic
    df = pd.read_hdf(args.traffic_file)
    Traffic = df.values
    # train/val/test 
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    # X, Y 
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # spatial embedding 
    f = open(args.SE_file, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]
        
    # temporal embedding 
    Time = df.index
    dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (5*60)
    timeofday = np.reshape(timeofday, newshape = (-1, 1))    
    Time = np.concatenate((dayofweek, timeofday), axis = -1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)
    
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)


def mae_los(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition = tf.math.is_nan(mask), x = 0., y = mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss


def mse_los(pred, label):
    ''''''
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition = tf.math.is_nan(mask), x = 0., y = mask)
    
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss


def rmse_los(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition = tf.math.is_nan(mask), x = 0., y = mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss