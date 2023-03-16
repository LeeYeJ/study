import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test) =mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(64,(2,2),padding='same',input_shape=(28,28,1)))
model.add(MaxPooling2D()) # 중첩 안하고 있는것중에 가장 큰것으로 뽑음 디폴트가 (2,2) / ( 반띵 됨!!! )
model.add(Conv2D(filters=64,kernel_size=(2,2),activation='relu'))
model.add(Conv2D(32,2)) # = (2,2)/ 예를들어 (3,3)이면 3 써줌
model.add(Flatten())
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7), activation='relu')
model.add(Dense(10,activation='softmax'))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 64)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        16448
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 32)        8224
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0
_________________________________________________________________
dense (Dense)                (None, 10)                46090
=================================================================
Total params: 71,082
Trainable params: 71,082
Non-trainable params: 0
'''