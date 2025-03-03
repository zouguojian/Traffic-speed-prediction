import numpy as np
data = np.load("test.npz",allow_pickle=True)
x = data['x']
y = data['y']
x_offsets = data['x_offsets']
y_offsets = data['y_offsets']
print(x[:,0,0,0], max(x[:,0,0,0]), min(x[:,0,0,0]))