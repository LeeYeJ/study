# x는 1개 Y는 3개 (실무에는 거의 안나옴)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([range(10)])  # range함수는 크기  range(10) -> [0,1,2,3,4,5,6,7,8,9]
x = x.T #(10,1)

y= np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
             [9,8,7,6,5,4,3,2,1,0]]) #(3,10)
y=y.T #(10,3)

#[실습]
#예측:[[9]] -> 예측[[10,1.9,0]]

model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(3))

model.compile(loss ='mse', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=1)
 
loss = model.evaluate(x,y) #훈련 데이터를 평가에 쓰면 안된다. 그렇다면 전체 데이터에서 일부는 훈련 시키고 일부는 평가에 사용한다.!!!!
print('loss :' , loss)

result = model.predict([[9]])
print('[[9]]의 result :', result)


'''
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))

model.compile(loss ='mse', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=1)

Epoch 1000/1000
10/10 [==============================] - 0s 889us/step - loss: 0.0841
1/1 [==============================] - 0s 101ms/step - loss: 0.0532
loss : 0.053242277354002
1/1 [==============================] - 0s 79ms/step
[[9]]의 result : [[10.109468   1.99487    0.7248926]]
'''
