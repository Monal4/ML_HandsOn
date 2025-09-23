import numpy as np
from sklearn import preprocessing

a = np.array([1, 2, 3])

b = np.array([[1.1, 2.1, 3.1, 4.1], [5.1, 6.1, 7.1, 8.1]], 'float16')

print(a, '\n', b)

# print dimension
print('Dimensions of a', np.ndim(a), 'and b', np.ndim(b))

# print size
print('Shape of a', np.shape(a), 'and b', np.shape(b))

# print type
print('Type of a', a.dtype, 'and b', b.dtype)

# print bytes
print('Bytes of a', a.nbytes, 'and b', b.nbytes)

# 1 1 1 1 1
# 1 0 0 0 1
# 1 0 9 0 1
# 1 0 0 0 1
# 1 1 1 1 1

Matrix = np.ones((5, 5), 'int8')
print('\n\n\n Initial Matrix \n', Matrix)
Matrix[1:4, 1:2] = 0
Matrix[1:4, 3:4] = 0
Matrix[1:4:2, 2:3] = 0
Matrix[2:3, 2:3] = 9
print('Result \n', Matrix)

print('\n\nAlternate Approach')
Matrix_new = Matrix.__copy__()
print(Matrix_new)
Zeros = np.zeros((3, 3), 'int8')
Zeros[1:2, 1:2] = 99
Matrix_new[1:4, 1:4] = Zeros
print('Zeros matrix \n', Zeros)
print('Result \n', Matrix_new)

array = np.array([
    [19, 12],
    [11, 13],
    [14, 17]
])

print(preprocessing.MinMaxScaler((-2,2)).fit_transform(array).transpose())
