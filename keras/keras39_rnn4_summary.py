# #리뷰
# model= Sequential()
# model.add(SimpleRNN(16,input_shape=(5,1),activation='linear'))
# 기본값이 activation = 'tanh' 이기때문에 -1 ~ 1 사이의 값이 나온다 따라서 linear로 바꿔줬더니 값이 맞게 나왔음

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# 시계열은 y모름

x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]]) #[8,9,10]은 예측해야되니까 데이터로 못만들어

y = np.array([6,7,8,9,10])

print(x.shape,y.shape) # (7, 3) (7,)

# RNN구조는 3차원 / x의 shape = (행, 열, 몇개씩 훈련하는지!!!)

x = x.reshape(5,5,1) #[[[1],[2],[3]],[[2],[3],[4]].............]
print(x.shape) # (7, 3, 1)

#2.모델
model= Sequential() # [batch, timesteps, feature]
model.add(SimpleRNN(10,input_shape=(5,1),activation='linear')) # 행 빼고 나머지/ 다른 모델들도 마찬가지임 / 32는 아웃풋 노드의 갯수
# Rnn은 삼차원 받아서 이차원 출력한다.

# units * (feature + biases + units) = params

model.add(Dense(7))
model.add(Dense(1))

model.summary()

'''
param 갯수 = ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1(biases) * unit 개수)
                  10*10                               + 1*10                              + 10

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn (SimpleRNN)      (None, 10)                120

 dense (Dense)               (None, 7)                 77   

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 205
Trainable params: 205
Non-trainable params: 0

'''
