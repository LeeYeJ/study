# x는 3개 Y는 2개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([range(10), range(21,31), range(201,211)])  # range함수는 크기  range(10) -> [0,1,2,3,4,5,6,7,8,9]
x = x.T #(10,3)

y= np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]]) #(2,10)
y=y.T #(10,2)

#모델구성 (전치 데이터의 모델)
model = Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(2))


#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=500, batch_size=1)

#4.평가 예측
loss=model.evaluate(x,y)
print('loss :', loss)

result = model.predict([[9,30,210]])
print('[[9,30,210]]의 result :', result)

#[실습] 
#예측 : [[9.20.210]] -> 예상값 [[10,1.9]]
'''
#모델구성
model = Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(2))

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=500, batch_size=1)

#4.평가 예측
loss=model.evaluate(x,y)
print('loss :', loss)

result = model.predict([[9,30,210]])
print('result :', result)

Epoch 500/500
10/10 [==============================] - 0s 776us/step - loss: 2.0735e-09
1/1 [==============================] - 0s 91ms/step - loss: 8.3120e-10
loss : 8.312042720781676e-10
1/1 [==============================] - 0s 79ms/step
result : [[9.999984  1.8999693]]
'''