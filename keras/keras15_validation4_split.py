# 훈련 데이터(train)에 일부를 검증(val) 데이터로 사용한다 즉 트레인 검증 테스트 세가지로 나뉨.

from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#데이터
x=np.array(range(1,17)) #(10,)
y=np.array(range(1,17)) 

x_train=np.array([14,15,16])
y_train=np.array([14,15,16])

x_test=np.array([11,12,13])
y_test=np.array([11,12,13])

#실습 :: 잘라보자
#train_test_split로만 잘라라
#10:3:3


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
          validation_split=0.2) # 훈련때 validation_split=0.2 나눠줄수있음
#ctrl + 스페이스바 -> 예약어 볼수있음

#평가예측
loss=model.evaluate(x_test,y_test)
print('loss :',loss)

result=model.predict([17])
print('[17]의 예측값:',result)


