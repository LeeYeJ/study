from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1.데이터
x= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]) #데이터를 시각화한다. ->scatter, 그래프 그린다는 소리 (예측값에 선을 그어보자)
y= np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20]) 

x_train, x_test, y_train, y_test = train_test_split(x, y, #x는 x_train과 x_test로 분리되고, y는 y_train과 y_test 순서로! 분리된다.
     train_size=0.7, shuffle=True, random_state= 1234
)

#2.모델 구성
model=Sequential()
model.add(Dense(6, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train,epochs=2000, batch_size=1)

#4.평가 예측
loss=model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict=model.predict(x) # 전체 데이터를 넣어보자

import matplotlib.pyplot as plt #그림그릴때 쓰는 api 
#시각화
plt.scatter(x,y) #점으로 볼수있음
#plt.scatter(x, y_predict, color='red')
plt.plot(x, y_predict, color='red') # plot 선으로 볼수있음

plt.show()

# 수치로 나오는 것  -> 회귀(지금 하고 있는것) , 예를들어 0/1로 나뉜다거나 남자 여자 나뉘는거 -> 분류 




