
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

#LSTM의 입력으로는 3차원의 데이터가 필요하기 때문입니다. (data size, time_steps, features)
#2.모델
model= Sequential()             #   [batch, timesteps, feature] 
                                # = [batch, input_lengh, input_dim]
# model.add(LSTM(10,input_shape=(5,1),activation='linear')) # 행 빼고 나머지/ 다른 모델들도 마찬가지임 / 32는 아웃풋 노드의 갯수
model.add(LSTM(10,input_length=5,input_dim=1)) # 이렇게도 표현가능
# model.add(LSTM(10,input_dim=1,input_length=5)) # 약간 가독성 떨어짐 , 어쨌든 똑같아

model.add(Dense(7))
model.add(Dense(1))

model.summary()


