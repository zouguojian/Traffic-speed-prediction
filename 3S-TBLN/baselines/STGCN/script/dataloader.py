import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def load_adj(dataset_name):
    dataset_path = './data'
    adj = sp.load_npz(os.path.join(dataset_path, dataset_name, 'adj.npz'))
    adj = adj.tocsc()
    
    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'ningxia-yc':
        n_vertex = 108

    return adj, n_vertex

def load_dataset(args):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(args.traffic_file+args.dataset+'/', category + '.npz'), allow_pickle=True)
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        print(data['x_' + category][...,0].shape, data['y_' + category][...,0].shape)
    mean=data['x_train'][..., 0].mean()
    std=data['x_train'][..., 0].std()
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = (data['x_' + category][..., 0]-mean)/std

    return data['x_train'][..., 0], data['y_train'][..., 0], data['x_val'][..., 0], data['y_val'][..., 0], data['x_test'][..., 0], data['y_test'][..., 0], mean, std


def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    return train, val, test

def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)