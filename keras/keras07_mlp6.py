# x는 3개 Y는 3개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([range(10), range(21,31), range(201,211)])  # range함수는 크기  range(10) -> [0,1,2,3,4,5,6,7,8,9]
x = x.T #(10,3)

y= np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
             [9,8,7,6,5,4,3,2,1,0]]) #(3,10)
y=y.T #(10,3)

model = Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3))

model.compile(loss ='mae', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=1) # 가중치가 결정된다.

loss = model.evaluate(x,y) #fit한 데이터를 널어주어서 훈련 로스값과 같다.
print('loss :' , loss)

result = model.predict([[9,30,210]])
print('[[9,30,210]]의 result :', result)

'''
model = Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3))

model.compile(loss ='mae', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=1)

Epoch 999/1000
10/10 [==============================] - 0s 855us/step - loss: 0.0399
Epoch 1000/1000
10/10 [==============================] - 0s 886us/step - loss: 0.0717
1/1 [==============================] - 0s 109ms/step - loss: 0.0533
loss : 0.05328764766454697
1/1 [==============================] - 0s 75ms/step
[[9,30,210]]의 result : [[9.977938   1.9249856  0.12519898]]
'''