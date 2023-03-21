# #리뷰
# model= Sequential()
# model.add(SimpleRNN(16,input_shape=(5,1),activation='linear'))
# 기본값이 activation = 'tanh' 이기때문에 -1 ~ 1 사이의 값이 나온다 따라서 linear로 바꿔줬더니 값이 맞게 나왔음

# vs코드 창 두개 띄워서 각각 gpu, cpu로 돌릴수있음(모델 동시에 돌리기 가능~ 성능만 받쳐준다면..)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# 시계열은 y모름

x = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]]) #[8,9,10]은 예측해야되니까 데이터로 못만들어

y = np.array([6,7,8,9,10])

print(x.shape,y.shape) # (7, 3) (7,)

# RNN구조는 3차원 / x의 shape = (행, 열, 몇개씩 훈련하는지!!!)

x = x.reshape(5,5,1) #[[[1],[2],[3]],[[2],[3],[4]].............]
print(x.shape) # (7, 3, 1)

#2.모델
model= Sequential()
model.add(SimpleRNN(64,input_shape=(5,1),activation='linear')) # 행 빼고 나머지/ 다른 모델들도 마찬가지임 / 32는 아웃풋 노드의 갯수
# model.add(Dense(16,activation='relu'))
# model.add(Dense(8))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(8))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
import time
start_time = time.time()
model.fit(x,y,epochs=10000)
end_time= time.time()
#4.평가예측
loss = model.evaluate(x,y)
x_predict = np.array([6,7,8,9,10]).reshape(1,5,1) #나올 데이터 한개 3,1은 똑같아/ [[[7],[8],[9],[10]]]
print(x_predict.shape)

result = model.predict(x_predict)
print('loss : ', loss)
print('[6,7,8,9,10]의 결과값 :', result)
print('걸린 시간은 :', round(end_time-start_time,2))

'''
Epoch 1000/1000
1/1 [==============================] - 0s 2ms/step - loss: 2.7285e-13
1/1 [==============================] - 0s 185ms/step - loss: 2.7285e-13
(1, 5, 1)
loss :  2.728484213739002e-13
[6,7,8,9,10]의 결과값 : [[11.]]

cpu로 돌렸을때 -> 이 모델에선 씨피유가 빠르지만 데이터가 많아질 경우 지피유가 빠를것이니 확인하고 쓰라~
Epoch 1000/1000
1/1 [==============================] - 0s 2ms/step - loss: 4.5475e-13
1/1 [==============================] - 0s 162ms/step - loss: 4.5475e-13
(1, 5, 1)
loss :  4.547473508864641e-13
[6,7,8,9,10]의 결과값 : [[11.000001]]
걸린 시간은 : 4.53
'''
