# 잡음 제거
# x로 x 룬련 -> 노이즈나 필요 없는 특성은 사라진다.
# 데이터의 특징을 추출하거나 차원을 축소
# 비지도 학습
# 생성 모델류는 loss에 너어무 의존하면 안됨 눈으로 봐야돼

import numpy as np
from tensorflow.keras.datasets import mnist

# 이미지로 이미지 찾는거니까 y 필요없음 그래서 x만 땡겨서 _로 표시
(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float32')/255. 
x_test = x_test.reshape(10000,784).astype('float32')/255.

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape = (784,))
encoded = Dense(64, activation='relu')(input_img) # 특성값을 축소 시킴 ( 큰 특성은 크게 남고 작은 특성은 소실됨 )

# decoded = Dense(784, activation='linear')(encoded) # ( 재구성! )
# decoded = Dense(784, activation='sigmoid')(encoded) 
# decoded = Dense(784, activation='relu')(encoded) 
decoded = Dense(784, activation='tanh')(encoded) 

autoencoder = Model(input_img, decoded) # 모델의 범위

autoencoder.summary() 

# autoencoder의 문제점 -> 특성이 강하지 않은 사진은 특성이 아닌 부분들을 제거하다보니 (경계선이) 흐려지는 문제점이 있음

autoencoder.compile(optimizer = 'adam',loss='mse') # 각 셀 값의 비교를 해야하므로 acc 의미없음, loss 비교를 하자
# autoencoder.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['acc']) # sigmoid도 mse 써도됨 성능만 좋으면 돼 


autoencoder.fit(x_train, x_train, epochs =30, batch_size = 128, validation_split=0.2 ) # x_train으로 x_train을 훈련시킨다.


