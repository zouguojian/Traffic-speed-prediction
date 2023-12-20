# -- coding: utf-8 --
import pandas as pd
import csv
import numpy as np

def adjacent(adjacent_file=None, road_segments=108, target_file='adjacent.txt'):
    '''
    :return: [N, N]
    '''
    adjacent=pd.read_csv(adjacent_file, encoding='utf-8').values

    full_adjacent = np.zeros(shape=[road_segments, road_segments])
    full_adjacent[:] = np.inf
    for pair in adjacent:
        full_adjacent[pair[0],pair[1]]=1

    # Calculates the standard deviation as theta.
    distances = full_adjacent[~np.isinf(full_adjacent)].flatten()
    std = distances.std()
    full_adjacent = np.exp(-np.square(full_adjacent / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    full_adjacent[full_adjacent < 0.1] = 0

    np.savez(target_file, data=full_adjacent)

if __name__=='__main__':
    print('hello')
    # 生成三元组形式的邻接矩阵
    adjacent(adjacent_file='adjacent.csv', road_segments=108,target_file='adjacent.npz')

    print('finished')