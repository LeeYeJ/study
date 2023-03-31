# 모델에서 리쉐잎 해는주는 방법~

from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Input,Dropout, LSTM,Conv1D,Flatten,Conv2D,MaxPooling2D,Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import accuracy_score


#실습

#1.데이터
# reshape 해주자~ cnn 성능보다 좋게 만들어라~
(x_train,y_train),(x_test,y_test) =mnist.load_data()
print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,) (10000,)
print()

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

x_train= x_train.reshape(60000,28,28,1)/255. # 리쉐잎과 스케일러 동시에.
x_test= x_test.reshape(10000,28,28,1)/255.
print(x_train.shape, x_test.shape) #(60000, 784) (10000, 784)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3),padding='same',input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Reshape(target_shape=(25,10))) #target_shape 어떻게 바꿀것인지/
model.add(Conv1D(10,3,padding='same'))
model.add(LSTM(784))
model.add(Reshape(target_shape=(28,28,1)))
model.add(Conv2D(32,3,padding='same'))
# model.add(Reshape(target_shape=(28*28))) # Reshape를 Flatten처럼 쓸순없다 에러남 이차원 안됨
model.add(Dense(10,activation='softmax'))


input1 = Input(shape=(28,28,1),name='H1') # 스칼라 13개 벡터 1개
conv1 = Conv2D(64,name='H2')(input1) #(input1) 이 레이어가 어디서 왔는지
maxp1 = MaxPooling2D()(conv1)
flat1 = Flatten()(maxp1)
resh1 = Reshape(target_shape=(25,10))(flat1)
conv2 = Conv1D(10,3,padding='same')(resh1)
lstm1 = LSTM(784)(conv2)
resh2 = Reshape(target_shape=(28,28,1))(lstm1)
conv3 = Conv2D(32,3,padding='same')(resh2)
output1 = Dense(10,activation='softmax',name='H5')(conv3)
model = Model(inputs = input1,outputs=output1) # 시작과 끝을 명시 / 모델 정의 제일 아래서

model.summary()