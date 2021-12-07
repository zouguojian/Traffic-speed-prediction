# -- coding: utf-8 --

import tensorflow as tf
import scipy.sparse as sp
import pandas as pd
import numpy as np

def get_position(num_roads=49):
    '''
    :return: shape is [1, 49]
    49 represents the numbers of road
    '''
    return np.array([[i for i in range(num_roads)]], dtype=np.int32)

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
    example [[1,0,0],[0,1,0],[0,0,1]]
    :return:
    '''

    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    print('adj_normalized shape is : ', adj_normalized.shape)

    return sparse_to_tuple(adj_normalized)

def adjecent(adj_file='adjacent.csv', num_roads=49):
    '''
    :return: adj matrix
    '''
    data = pd.read_csv(filepath_or_buffer=adj_file)
    adj = np.zeros(shape=[num_roads, num_roads])
    for line in data[['src_FID', 'nbr_FID']].values:
        adj[line[0]][line[1]] = 1
    return adj

def get_max_min():
    '''
    :return: the max and min value of input features
    '''

    min_list=[106007, 106007, 2020, 8, 31, 23, 148.021, 360]
    max_list=[2005, 2005, 2020, 5, 1, 0, 0.0, 0]

    return max_list, min_list

def normalization(data=None):
    '''
    :param data:
    :return:
    '''
    min_value=0.000000000001
    max,min=get_max_min()
    shape=data.shape

    for i in range(1,3):
        data[:,shape[-1]-i]=(data[:,shape[-1]-i] - np.array(min[shape[-1]-i])) / \
                            (np.array(max[shape[-1]-i]) - np.array(min[shape[-1]-i]+ min_value))

    return data

def re_recover(a, max=148.021, min=0.0):
    '''
    :param a: represent a list or one-dimension array.
    :param max: max number
    :param min: min number
    :return: op-normalization list or one-dimension array.
    '''
    return [num * (max - min) + min for num in a]

def feature_split(data, time_size=6, prediction_size=1, num_roads=49):
    '''
    :return: features: [batch size * input time size, num roads, features],
            days: [input time size + prediction time size, num roads],
            hours: [input time size + prediction time size, num roads].
    '''

    low=int(data.shape[0]//num_roads * 0.9) * num_roads

    (features, days, hours) = (data[:time_size * num_roads, 6:7],
                               data[:(time_size + prediction_size) * num_roads, 4],
                               data[:(time_size + prediction_size) * num_roads, 5])
    features=np.reshape(features,newshape=[time_size, num_roads, 1])
    days=np.reshape(days,newshape=[time_size+prediction_size,num_roads])
    hours=np.reshape(hours,newshape=[time_size+prediction_size,num_roads])

    return features,days,hours

def prediction(features=None, days=None, hours=None, num_roads=49):
    '''
    input_features shape is [batch size * input time size, num roads, features],
    for example,(1, 49,1), dtype: float64.

    input_position shape is [1, num roads],
    for example,(1, 49), dtype: int32.

    input_day shape is [input time size + prediction time size, num roads],
    for example,(7, 49), dtype: int32. 6 + 1 = 7

    input_hour shape is [input time size + prediction time size, num roads],
    for example,(7, 49), dtype: int32. 6 + 1 = 7

    input_indices shape is [None, 2], dtype : int 32.

    input_values shape is [None], dtype: float64.

    input_dense_shape shape is (num roads, num roads)
    :return:    pred shape is [batch size, num roads, prediction time size],
                example  (1, 49, 1), dtype: float.
    '''

    with tf.gfile.GFile('model.pb', 'rb') as fgraph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fgraph.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

        input_postion = graph.get_tensor_by_name('input_position:0')
        input_day = graph.get_tensor_by_name('input_day:0')
        input_hour = graph.get_tensor_by_name('input_hour:0')
        input_indices = graph.get_tensor_by_name('input_indices:0')
        input_values = graph.get_tensor_by_name('input_values:0')
        input_dense_shape = graph.get_tensor_by_name('input_dense_shape:0')
        input_features = graph.get_tensor_by_name('input_features:0')

        pred = graph.get_tensor_by_name('output_y:0')

        position=get_position(num_roads)
        adj=adjecent()
        adj=preprocess_adj(adj)

        # print(support)
        print(position.shape, position.dtype)
        print(days.shape,days.dtype)
        print(hours.shape,hours.dtype)
        print(features.shape,features.dtype)
        print(adj[0].shape, adj[0].dtype)
        print(adj[1].shape, adj[1].dtype)
        print(adj[2])

        sess = tf.Session(graph=graph)
        feed={input_postion:position,
              input_day:days,
              input_hour:hours,
              input_features:features,
              input_indices:adj[0],
              input_values:adj[1],
              input_dense_shape:adj[2]}

        scores = sess.run(pred, feed_dict=feed)
    return scores

'''input example'''
train_data=pd.read_csv('train.csv', encoding='utf-8')
features, days, hours=feature_split(data=train_data.values)

print(features.shape, days.shape, hours.shape)

# features=np.random.random([6,49,1])
# days=np.random.randint(low=1,high=20,size=[7, 49],dtype=np.int32)
# hours = np.random.randint(low=0, high=20, size=[7, 49],dtype=np.int32)

pres=prediction(features=features,days=days,hours=hours, num_roads=49)
pres=pres.transpose([1, 0, 2])

pres=np.array([re_recover(np.reshape(road_label, [-1])) for road_label in pres])

pres_dict={}
pres_dict={(int(train_data.values[i][0]), int(train_data.values[i][1])): pres[i] for i in range(49)}

for pres_d in pres_dict:
    print(pres_d, pres_dict[pres_d])