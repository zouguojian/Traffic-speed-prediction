# -- coding: utf-8 --

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    print(x.shape)
    print(y.shape)
    print(tx.shape)
    print(allx.shape)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    print(rowsum.shape)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    print(r_inv.shape)
    r_mat_inv = sp.diags(r_inv)
    print(r_mat_inv.shape)
    features = r_mat_inv.dot(features)
    print('features shape is : ',features.shape)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    '''
    :param adj: Symmetrically normalize adjacency matrix
    :return:
    '''
    adj = sp.coo_matrix(adj) # 转化为稀疏矩阵表示的形式
    rowsum = np.array(adj.sum(1)) # 原连接矩阵每一行的元素和
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() #先根号，再求倒数，然后flatten返回一个折叠成一维的数组
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. #
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    '''
    :param adj:  A=A+E, and then to normalize the the adj matrix,
    preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
    :return:
    '''
    # 邻接矩阵 加上 单位矩阵
    '''
    [[1,0,0],[0,1,0],[0,0,1]]
    '''
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    print('adj_normalized shape is : ', adj_normalized.shape)

    return sparse_to_tuple(adj_normalized)



def construct_feed_dict(features, adj, labels,  day, hour, minute, placeholders, site_num=66):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['position']: np.array([[i for i in range(site_num)]],dtype=np.int32)})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['day']: day})
    feed_dict.update({placeholders['hour']: hour})
    feed_dict.update({placeholders['minute']: minute})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['indices_i']: adj[0]})
    feed_dict.update({placeholders['values_i']: adj[1]})
    feed_dict.update({placeholders['dense_shape_i']: adj[2]})
    # feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def accuracy(label, predict):
    '''
    :param label: represents the observed value
    :param predict: represents the predicted value
    :param epoch:
    :param steps:
    :return:
    '''
    error = label - predict
    average_error = np.mean(np.fabs(error.astype(float)))
    print("MAE is : %.6f" % (average_error))

    rmse_error = np.sqrt(np.mean(np.square(label - predict)))
    print("RMSE is : %.6f" % (rmse_error))

    cor = np.mean(np.multiply((label - np.mean(label)),
                              (predict - np.mean(predict)))) / (np.std(predict) * np.std(label))
    print('R is: %.6f' % (cor))

    # mask = label != 0
    # mape =np.mean(np.fabs((label[mask] - predict[mask]) / label[mask]))*100.0
    # mape=np.mean(np.fabs((label - predict) / label)) * 100.0
    # print('mape is: %.6f %' % (mape))
    sse = np.sum((label - predict) ** 2)
    sst = np.sum((label - np.mean(label)) ** 2)
    R2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
    print('R^$2$ is: %.6f' % (R2))

    return average_error, rmse_error, cor, R2

