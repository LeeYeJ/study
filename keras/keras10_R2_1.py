from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1.데이터
x= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]) #데이터를 시각화한다. ->scatter, 그래프 그린다는 소리 (예측값에 선을 그어보자)
y= np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20]) 

x_train, x_test, y_train, y_test = train_test_split(x, y, #x는 x_train과 x_test로 분리되고, y는 y_train과 y_test 순서로! 분리된다.
     train_size=0.9, shuffle=True, random_state= 1234
)

#2.모델 구성
model=Sequential()
model.add(Dense(6, input_dim=1))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train,epochs=700, batch_size=1)

#4.평가 예측
loss=model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict=model.predict(x_test) # 훈련 안시킨 데이터에서 예측하자 아래

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # 원값과 예측값이 얼마나 맞는지 확인할 수 있다. / 얘도 훈련안한 y_test로 확인해보자 (내신,수능 비교)
print('r2스코어 :', r2) # 값은 1과 가까울 수록 좋다.

# r2와 loss의 값이 엉키면 loss로 판단한다. 거의 절대적인 값이기 때문에

'''
Epoch 700/700
18/18 [==============================] - 0s 937us/step - loss: 2.4225
1/1 [==============================] - 0s 178ms/step - loss: 0.9669
loss :  0.9668638706207275
1/1 [==============================] - 0s 65ms/step
r2스코어 : 0.9739477488706364
'''
