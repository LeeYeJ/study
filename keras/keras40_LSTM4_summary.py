# #리뷰
# model= Sequential()
# model.add(SimpleRNN(16,input_shape=(5,1),activation='linear'))
# 기본값이 activation = 'tanh' 이기때문에 -1 ~ 1 사이의 값이 나온다 따라서 linear로 바꿔줬더니 값이 맞게 나왔음

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

#LSTM의 입력으로는 3차원의 데이터가 필요하기 때문입니다. (data size, time_steps, features)
#2.모델
model= Sequential() # [batch, timesteps, feature]
model.add(LSTM(10,input_shape=(5,1),activation='linear')) # 행 빼고 나머지/ 다른 모델들도 마찬가지임 / 32는 아웃풋 노드의 갯수
# Rnn은 삼차원 받아서 이차원 출력한다.

# units * (feature + biases + units) = params

model.add(Dense(7))
model.add(Dense(1))

model.summary()

'''
블로그 참고
https://blog.naver.com/winddori2002/221992543837

params = dim(W)+dim(V)+dim(U) = n*n + kn + nm

# n - dimension of hidden layer

# k - dimension of output layer

# m - dimension of input layer
'''

'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 10)                480    # simplernn 연산량의 4배이다. 

 dense (Dense)               (None, 7)                 77

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 565
Trainable params: 565
Non-trainable params: 0

'''
