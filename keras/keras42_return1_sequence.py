#LSTM에서 던져주는 값이 2차원이 아닌 3차원으로 던져줘서 그 담에 Dense가 아닌 LSTM을 엮어줄수있다.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])

y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape,y.shape) # (7, 3) (7,)

# RNN구조는 3차원 / x의 shape = (행, 열, 몇개씩 훈련하는지!!!)

x = x.reshape(13,3,1) #[[[1],[2],[3]],[[2],[3],[4]].............]
print(x.shape) # (7, 3, 1)

#2.모델
model= Sequential()
# model.add(LSTM(10,input_shape=(3,1)))
# model.add(LSTM(11))  #ValueError: Input 0 of layer "lstm_1" is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 32) # 차원 안맞음 고로 아래

model.add(LSTM(10,input_shape=(3,1),return_sequences=True)) # return_sequences=True 순서를 되돌려준다 LSTM들을 엮을수있음 차원을 3차원으로 던져줘서 # 원래는 3차원으로 들어가서 2차원으로 나옴
model.add(LSTM(11, return_sequences=True)) # LSTM과 GRU도 return_sequences=True를 써서 엮어줄수있음
model.add(GRU(12))
model.add(Dense(1))

model.summary()

#3.컴파일 훈련
# model.compile(loss='mse',optimizer='adam')
# model.fit(x,y,epochs=500)

# #4.평가예측
# loss = model.evaluate(x,y)
# x_predict = np.array([8,9,10]).reshape(1,3,1) #나올 데이터 한개 3,1은 똑같아/ [[[8],[9],[10]]]
# print(x_predict.shape)

# result = model.predict(x_predict)
# print('loss : ', loss)
# print('[8,9,10]의 결과값 :', result)

'''
loss :  4.4310503005981445
[8,9,10]의 결과값 : [[6.0028834]]
'''
'''
loss :  7.110954175004736e-05
[8,9,10]의 결과값 : [[10.866173]]
'''