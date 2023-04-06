from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

#데이터
(x_train,y_train),(x_test,y_test) = reuters.load_data(
    num_words=10000, test_split=0.2 #num_words=10000 상위 1만개(input_dim)
)
print(x_train)
print(x_train.shape, y_train.shape) # (8982,) (8982,)
print(x_test.shape,y_test.shape) # (2246,) (2246,)

print(len(x_train[0]), len(x_train[1])) #87 56
print(np.unique(y_train)) #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'>
print(type(x_train[0])) #  <class 'list'> 넘파이 안에 리스트가 들어있음

print('뉴스 기사의 최대 길이: ',max(len(i) for i in x_train)) # 뉴스 기사의 최대 길이:  2376
print('뉴스 기사의 평균 길이: ',sum(map(len, x_train))/len(x_train)) # 뉴스 기사의 평균 길이:  145.5398574927633

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train,padding='pre',maxlen=100,
                        truncating='pre') # 뒤에 100개 남기고 앞을 잘라버린다.
print(x_train.shape) #(8982, 100)

# 나머지 전처리
x_test = pad_sequences(x_test,padding='pre',maxlen=100,
                        truncating='pre') # 뒤에 100개 남기고 앞을 잘라버린다.
print(x_test.shape) #(2246, 100)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense

model= Sequential()
# model.add(Reshape(target_shape=(5,1), input_shape=(5,)))
model.add(Embedding(28,32,input_length=100))
# model.add(Bidirectional(LSTM(10,return_sequences=True),input_shape=(5,1))) #Bidirectional 혼자 못씀 양방향으로 쓰기위해/ RNN 모델을 래핑하는 형태로 써야
model.add(LSTM(10))
# model.add(Bidirectional(GRU(10)))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(8))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(16))
# model.add(Dense(8))
model.add(Dense(10))
# model.add(Dense(16))
model.add(Dense(1,activation='sigmoid'))

# 시작
model.compile(loss ='binary_crossentropy', optimizer ='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, batch_size=8)

acc = model.evaluate(x_test,y_test)[1] # -> loss 와 acc 값
print('acc :', acc)
