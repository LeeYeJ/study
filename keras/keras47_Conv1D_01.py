import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

#1. 데이터
#2. 모델 

model = Sequential()
# model.add(LSTM(10,input_shape=(3,1))) #토탈 파람스 :541
model.add(Conv1D(10,2, input_shape=(3,1)))  #토탈 파람스 :141
model.add(Conv1D(10,2))  #토탈 파람스 :301
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))

model.summary() # conv2d와 파람 계산 동일
