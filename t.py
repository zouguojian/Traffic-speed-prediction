import numpy as np

a=np.random.randint(low=1,high=100,size=[2, 4, 3])

print(a)

# b=np.reshape(a,newshape=[-1,3])
# print(b)


c=np.reshape(a,newshape=[-1, 2, 2, 3])
print(c)