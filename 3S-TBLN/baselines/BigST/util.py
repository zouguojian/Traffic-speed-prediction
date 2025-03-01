import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg

class DataLoader(object):
    def __init__(self, data, batch_size, input_length, output_length):
        self.seq_length_x = input_length
        self.seq_length_y = output_length
        self.y_start = 1
        self.batch_size = batch_size
        self.current_ind = 0
        self.x_offsets = np.sort(np.concatenate((np.arange(-(self.seq_length_x - 1), 1, 1),)))
        self.y_offsets = np.sort(np.arange(self.y_start, (self.seq_length_y + 1), 1))
        self.min_t = abs(min(self.x_offsets))
        self.max_t = abs(data.shape[0] - abs(max(self.y_offsets)))
        mod = (self.max_t-self.min_t) % batch_size
        if mod != 0:
            self.data = data[:-mod]
        else:
            self.data = data
        self.max_t = abs(self.data.shape[0] - abs(max(self.y_offsets)))
        self.permutation = [i for i in range(self.min_t, self.max_t)]

    def shuffle(self):
        self.permutation = np.random.permutation([i for i in range(self.min_t, self.max_t)])

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < len(self.permutation):
                if self.batch_size > 1:
                    x_batch = []
                    y_batch = []
                    for i in range(self.batch_size):  
                        x_i = self.data[self.permutation[self.current_ind+i] + self.x_offsets, ...]
                        y_i = self.data[self.permutation[self.current_ind+i] + self.y_offsets, ...]
                        x_batch.append(x_i)
                        y_batch.append(y_i)

                    x_batch = np.stack(x_batch, axis=0)
                    y_batch = np.stack(y_batch, axis=0)
                else:
                    x_batch = self.data[self.permutation[self.current_ind] + self.x_offsets, ...]
                    y_batch = self.data[self.permutation[self.current_ind] + self.y_offsets, ...]
                    x_batch = np.expand_dims(x_batch, axis=0)
                    y_batch = np.expand_dims(y_batch, axis=0)
                yield (x_batch, y_batch)
                self.current_ind += self.batch_size

        return _wrapper()

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
        

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
        

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
    

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()
    

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian
    

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()
    

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
    

def load_adj(adj_filename):
    adj_mx = np.load(adj_filename, allow_pickle=True)['data']
    # adj_mx = np.load(adj_filename)
    print('adj_mx: ', adj_mx.shape)
    adj = [asym_adj(adj_mx)]
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'), allow_pickle=True)
        data['x_' + category] = cat_data['x'].astype(float)
        data['y_' + category] = cat_data['y'].astype(float)
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data
