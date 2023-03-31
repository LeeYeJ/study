# LSTM이 Conv1보다 성능이 더 좋지만 거의 유사하다 (Conv1이 시간 덜걸림) -> conv2보다 conv1이 차원 하나 줄음 쉐잎 맞춰주자~
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU,Bidirectional

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])

y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(13,3,1)
print(x.shape)

model= Sequential()
model.add(Bidirectional(GRU(16,activation='linear'),input_shape=(3,1))) # 행 빼고 나머지/ 다른 모델들도 마찬가지임 / 32는 아웃풋 노드의 갯수
model.add(Dense(16,activation='relu'))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(10,activation='relu'))
model.add(Dense(16))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
import time
start_time = time.time()
model.fit(x,y,epochs=100)
end_time= time.time()

#4.평가예측
loss = model.evaluate(x,y)
x_predict = np.array([50,60,70]).reshape(1,3,1) 
print(x_predict.shape)

result = model.predict(x_predict)
print('loss : ', loss)
print('[50,60,70]의 결과값 :', result)
print('걸린 시간은 :', round(end_time-start_time,2))

'''
GRU만 쓴거

loss :  1.5210658311843872
[50,60,70]의 결과값 : [[84.47385]]
걸린 시간은 : 6.39

Bidirectional(GRU) 써준거

'''
