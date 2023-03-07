# 훈련 데이터(train)에 일부를 검증(val) 데이터로 사용한다 즉 트레인 검증 테스트 세가지로 나뉨.

from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#데이터
x=np.array(range(1,17)) #(10,)
y=np.array(range(1,17)) 

# x_val=np.array([14,15,16])
# y_val=np.array([14,15,16])

# x_test=np.array([11,12,13])
# y_test=np.array([11,12,13])

#실습 :: 잘라보자
#train_test_split로만 잘라라
#10:3:3
x_train, x_test, y_train, y_test= train_test_split(x, y, train_size=0.625, random_state=1)

x_test, x_val, y_test, y_val= train_test_split(x_test, y_test, train_size=0.5, random_state=2)

print(x_train,x_test,x_val)

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

'''
validation_data(x_val, y_val) = 검증 데이터(validation data)를 사용합니다. 
일반적으로 검증 데이터를 사용하면 각 에포크마다 검증 데이터의 정확도나 오차를 함께 출력하는데, 
이 정확도는 훈련이 잘 되고 있는지를 보여줄 뿐이며 실제로 모델이 검증 데이터를 학습하지는 않습니다. 
검증 데이터의 오차(loss)가 낮아지다가 높아지기 시작하면 이는 과적합(overfitting)의 신호입니다.

validation_split = validation_data와 동일하게 검증 데이터를 사용하기 위한 용도로 validation_data 대신 사용할 수 있습니다.
검증 데이터를 지정하는 것이 아니라 훈련 데이터와 훈련 데이터의 레이블인 X_train과 y_train에서 
일정 비율 분리하여 이를 검증 데이터로 사용합니다.

https://wikidocs.net/32105
'''

#평가예측
loss=model.evaluate(x_test,y_test)
print('loss :',loss)

result=model.predict([17])
print('[17]의 예측값:',result)


