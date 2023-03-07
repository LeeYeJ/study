# 훈련 데이터(train)에 일부를 검증(val) 데이터로 사용한다 즉 트레인 검증 테스트 세가지로 나뉨.

from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
import numpy as np

#데이터
x_train=np.array(range(1,17)) #(10,)
y_train=np.array(range(1,17)) 

x_val=x_train[13:]
y_val=y_train[13:]
print(x_val,y_val) #[14 15 16] [14 15 16]

x_test=x_train[10:13]
y_test=y_train[10:13]
print(x_test,y_test) #[11 12 13] [11 12 13]

#실습 :: 잘라보자
x_val=np.array
y_val=np.array

x_test=np.array([11,12,13])
y_test=np.array([11,12,13])

#모델
model=Sequential()
model.add(Dense(5,activation='linear',input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1,
          validation_data=(x_val,y_val)) #훈련하고 검증하고의 반복  val_loss: 0.4519 검증 로스값도 나옴

#평가예측
loss=model.evaluate(x_test,y_test)
print('loss :',loss)

result=model.predict([17])
print('[17]의 예측값:',result)


