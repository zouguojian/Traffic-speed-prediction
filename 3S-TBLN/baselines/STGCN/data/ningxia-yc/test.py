# -- coding: utf-8 --

'''
将密集矩阵转化为稀疏矩阵，按照以下步骤进行实现
'''
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse
import scipy.sparse as sp

cat_data = np.load('adjacent.npz', allow_pickle=True)

print(list(cat_data.keys()))

dense_metr = cat_data['data']
adj_sp=scipy.sparse.csr_matrix(dense_metr)
sp.save_npz('adj.npz',matrix=adj_sp)
