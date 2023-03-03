#전체 데이터에서 일부는 훈련 시키고 일부는 평가에 사용한다.!!!! (훈련한 데이터를 평가하는 것은 그렇게 의미있는 행동은 아니잖아)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([10,9,8,7,6,5,4,3,2,1,]) #뒤에 ,는 뒤에 더 있을 수도 있고 없을 수도 있다는 뜻이다. 그냥 상관없다는 뜻

#print(x)
#print(y)

x_train=np.array([1,2,3,4,5,6,7])
y_train=np.array([1,2,3,4,5,6,7])
x_test=np.array([8,9,10])
y_test=np.array([8,9,10])

#2.모델 구성
model=Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=50,batch_size=1) #가중치가 결정됨

#평가 훈련
loss= model.evaluate(x_test,y_test)
print('loss :', loss)

result=model.predict([11])
print('[11]의 예측값 :', result)

'''
Epoch 50/50
7/7 [==============================] - 0s 913us/step - loss: 2.2154e-04
1/1 [==============================] - 0s 96ms/step - loss: 9.8224e-04
loss : 0.0009822356514632702
1/1 [==============================] - 0s 77ms/step
[11]의 예측값 : [[10.955775]]   x,y train 배열에 의해 11 근사치가 나와야함
'''
'''
로스는 낮은데 결과값이 맞지 않는 이유?
이것의 해답이 아마 랜덤 데이터 분리인듯 왜냐하면 데이터 범위를 무자르듯 자르면 직선이기 때문에 오차가 점점 커질수 있음
따라서 랜덤하게 데이터를 추출해서 훈련 평가하는 것이 좋음
'''