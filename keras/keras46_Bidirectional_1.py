# 양방향 LSTM

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM, GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# 시계열은 y모름

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]]) #[8,9,10]은 예측해야되니까 데이터로 못만들어

y = np.array([4,5,6,7,8,9,10])

print(x.shape,y.shape) # (7, 3) (7,)
x = x.reshape(7,3,1) #[[[1],[2],[3]],[[2],[3],[4]].............]
print(x.shape) # (7, 3, 1)

#모델구성
model= Sequential()
model.add(Bidirectional(LSTM(10,return_sequences=True),input_shape=(3,1))) #Bidirectional 혼자 못씀 양방향으로 쓰기위해/ RNN 모델을 래핑하는 형태로 써야
model.add(LSTM(10,return_sequences=True))
model.add(Bidirectional(GRU(10)))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 bidirectional (Bidirectiona  (None, 3, 20)            960
 l)

 lstm_1 (LSTM)               (None, 3, 10)             1240

 bidirectional_1 (Bidirectio  (None, 20)               1320
 nal)

=================================================================
Total params: 3,520
Trainable params: 3,520
Non-trainable params: 0
'''
