# #리뷰
# GRU는 LSTM 연산량 중에 중복되는 부분은 삭제했다.
# GRU 정리해보기
#https://huidea.tistory.com/237

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM,GRU
from tensorflow.python.keras.callbacks import EarlyStopping


#2.모델
model= Sequential() # [batch, timesteps, feature]
model.add(GRU(10,input_shape=(5,1))) # 행 빼고 나머지/ 다른 모델들도 마찬가지임 / 32는 아웃풋 노드의 갯수
model.add(Dense(7))
model.add(Dense(1))

model.summary()
'''
3 * (다음 노드 수^2 +  다음 노드 수 * Shape 의 feature + 다음 노드수 )
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 gru (GRU)                   (None, 10)                390

 dense (Dense)               (None, 7)                 77

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 475000000000000000000000000000000000000000000000000000000
Trainable params: 475
Non-trainable params: 0
'''


