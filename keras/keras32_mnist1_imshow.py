import numpy as np
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) =mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) # 흑백은 (1은) 생략 가능
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

print(x_train)
print(y_train)
print('===========')
print(y_test)
# print(x_train[0])
# print(y_train[333])

# import matplotlib.pyplot as plt
# plt.imshow(x_train[333],'gray')
# plt.show()