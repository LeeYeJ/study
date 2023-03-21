# Rnn에서 제일 성능 좋은게 엘에스티엠

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# 시계열은 y모름

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]]) #[8,9,10]은 예측해야되니까 데이터로 못만들어

y = np.array([4,5,6,7,8,9,10])

print(x.shape,y.shape) # (7, 3) (7,)

# RNN구조는 3차원 / x의 shape = (행, 열, 몇개씩 훈련하는지!!!)

x = x.reshape(7,3,1) #[[[1],[2],[3]],[[2],[3],[4]].............]
print(x.shape) # (7, 3, 1)

#2.모델
model= Sequential()
model.add(LSTM(16,input_shape=(3,1))) # 행 빼고 나머지/ 다른 모델들도 마찬가지임 / 32는 아웃풋 노드의 갯수
model.add(Dense(16,activation='relu'))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(10,activation='relu'))
model.add(Dense(16))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=500)

#4.평가예측
loss = model.evaluate(x,y)
x_predict = np.array([8,9,10]).reshape(1,3,1) #나올 데이터 한개 3,1은 똑같아/ [[[8],[9],[10]]]
print(x_predict.shape)

result = model.predict(x_predict)
print('loss : ', loss)
print('[8,9,10]의 결과값 :', result)

'''
loss :  4.4310503005981445
[8,9,10]의 결과값 : [[6.0028834]]
'''
'''
loss :  7.110954175004736e-05
[8,9,10]의 결과값 : [[10.866173]]
'''