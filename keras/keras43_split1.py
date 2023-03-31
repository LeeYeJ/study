# # 시계열은 데이터를 어떻게 자르느냐가 중요하다
# #https://www.w3schools.com/python/numpy/numpy_array_slicing.asp 참고
# import numpy as np

# dataset = np.array(range(1,11))
# timesteps =5 # 5개씩 잘라

# def split_x(dataset, timesteps):
#     aaa=[]
#     for i in range(len(dataset) - timesteps + 1):
#         subset = dataset[i:(i + timesteps)]
#         aaa.append(subset)
#     return np.array(aaa)

# bbb = split_x(dataset, timesteps)
# print(bbb)
# '''
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
# '''
# print(bbb[:,:-1])
# print(bbb[:,-1:])
# # print(bbb.shape) # (6, 5)

import numpy as np

dataset = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
timesteps =2 # 5개씩 잘라

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i:(i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(dataset, timesteps)

print(bbb)
'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
'''
# print(bbb[:,:-1])
# print(bbb[:,-1:])
# print(bbb.shape) # (6, 5)
