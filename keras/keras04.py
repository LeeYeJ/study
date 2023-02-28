import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

#2.모델구성

model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=495)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss :', loss)

result = model.predict([6])
print('[6]의 예측값은 : ', result)

#loss : 0.40338340401649475
#1/1 [==============================] - 0s 77ms/step
#[6]의 예측값은 :  [[6.0056105]]