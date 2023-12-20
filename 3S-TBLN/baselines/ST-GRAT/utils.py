import numpy as np
import pandas as pd
import os
import pickle

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def load_graph_adj(pkl_filename):
    adj_mx = np.load(pkl_filename)['data']
    return adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return adj_mx

# metric
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def load_dataset(args):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(args.traffic_file+args.name+'/', category + '.npz'), allow_pickle=True)
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        print(data['x_' + category][...,0].shape, data['y_' + category].shape)
    mean=data['x_train'][..., 0].mean()
    std=data['x_train'][..., 0].std()
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = (data['x_' + category][..., 0]-mean)/std

    # spatial embedding
    f = open(args.traffic_file+args.name+'/'+args.SE_file, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]

    # temporal embedding
    trainTE = None
    valTE = None
    testTE = None

    return data['x_train'][..., 0], trainTE, data['y_train'][..., 0], data['x_val'][..., 0], valTE, data['y_val'][..., 0], data['x_test'][..., 0], testTE, data['y_test'][..., 0], SE, mean, std

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
                // Time.freq.delta.total_seconds()
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

def loadPEMSData(args):
    # Traffic
    Traffic = np.squeeze(np.load(args.traffic_file)['data'], -1)
    # Traffic = load_partition(args, Traffic)
    print(Traffic.shape)
    # train/val/test 
    num_step = Traffic.shape[0]
    # padding = [[0 for i in range(num_step)] for i in range(1)]
    # padding = np.stack(padding, -1).astype(np.float32)
    # Traffic = np.concatenate([Traffic, padding], -1)
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

    trainY = (trainY - mean) / std
    valY = (valY - mean) / std
    testY = (testY - mean) / std

    # spatial embedding 
    SE = np.load(args.SE_file).astype(np.float32)
    # paddingSE = [[0 for i in range(1)] for i in range(SE.shape[1])]
    # paddingSE = np.stack(paddingSE, -1)
    # SE = np.concatenate([SE,paddingSE], 0).astype(np.float32)
    # f = open(args.SE_file, mode = 'r')
    # lines = f.readlines()
    # temp = lines[0].split(' ')
    # N, dims = int(temp[0]), int(temp[1])
    # SE = np.zeros(shape = (N, dims), dtype = np.float32)
    # for line in lines[1 :]:
    #     temp = line.split(' ')
    #     index = int(temp[0])
    #     SE[index] = temp[1 :]
        
    # temporal embedding
    trainTE = None
    valTE = None
    testTE = None
    
    return trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std
