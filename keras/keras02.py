# 1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델 구성
import tensorflow as tf
from tensorflow.keras.models import Sequential # sequential 가져와라
from tensorflow.keras.layers import Dense  # Dense 기져와라

model = Sequential() 
model.add(Dense(1, input_dim=1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # loss는 mse로 연산하라 최적은 adam을 써라 
model.fit( x, y, epochs=100 ) # fit (훈련을 시켜라), x와 y 데이터를 넣고 ,훈련(epochs) 1000번 시킴

#loss:0.0097


