# 넘파이 리스트 슬라이싱 
'''
전체 데이터에서 일부는 훈련 시키고 일부는 평가에 사용한다.!!!! (훈련한 데이터를 평가하는 것은 그렇게 의미있는 행동은 아니잖아)
평가는 훈련데이터 범위 안의 데이터를 사용하되 훈련 데이터를 사용해선 안된다.
훈련과 평가 데이터는 분리를 하되 가능한 데이터는 전체적으로 훈련한다. 그러기 위해선 아래
셔플 개념은 데이터를 랜덤하게 섞어서 뽑아쓴다. 예를들면 셔플해서 70프로의 데이터를 훈련시킨다.
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([10,9,8,7,6,5,4,3,2,1,]) 

#[실습] 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라
x_train = x[:7]
y_train = y[:7]
x_test = x[7:]
y_test = y[7:]

print(x_train.shape,x_test.shape) #(7,) (3,)
print(y_train.shape,y_test.shape) #(7,) (3,)

model=Sequential()
model.add(Dense(6,input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1)

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

result=model.predict([11])
print('result :', result)
'''

model=Sequential()
model.add(Dense(6,input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1)


Epoch 100/100
7/7 [==============================] - 0s 831us/step - loss: 0.0012
1/1 [==============================] - 0s 93ms/step - loss: 0.0026
loss : 0.0026016277261078358
1/1 [==============================] - 0s 85ms/step
result : [[0.07637118]]
'''

