from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd

(x_train,y_train),(x_test,y_test) = imdb.load_data(
    num_words=10000
)

print(x_train)
print(y_train)
print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(np.unique(y_train,return_counts=True)) #(array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))

# 판다스에서는 value_counts 사용하면 됨

print(pd.value_counts(y_train))
'''
1    12500
0    12500
'''

print('영화평의 최대 길이: ',max(len(i) for i in x_train))  # 2494
print('영화평의 평균 길이: ',sum(map(len, x_train))/len(x_train)) # 238.71364

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train,padding='pre',maxlen=250,
                        truncating='pre') # 뒤에 100개 남기고 앞을 잘라버린다.
print(x_train.shape) #(8982, 100)

# 나머지 전처리
x_test = pad_sequences(x_test,padding='pre',maxlen=250,
                        truncating='pre') # 뒤에 100개 남기고 앞을 잘라버린다.
print(x_test.shape) #(2246, 100)

#실습

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM,Dense 

#2. 모델
model= Sequential()
# model.add(Reshape(target_shape=(5,1), input_shape=(5,)))
model.add(Embedding(10000,64,input_length=250))
# model.add(Bidirectional(LSTM(10,return_sequences=True),input_shape=(5,1))) #Bidirectional 혼자 못씀 양방향으로 쓰기위해/ RNN 모델을 래핑하는 형태로 써야
model.add(LSTM(10))
model.add(Dense(10))
# model.add(Dense(16))
model.add(Dense(1,activation='sigmoid'))


model.compile(loss ='binary_crossentropy', optimizer ='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, batch_size=8)

acc = model.evaluate(x_test,y_test)[1] # -> loss 와 acc 값
print('acc :', acc)