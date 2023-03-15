from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten # cnn 하겠다는것

model = Sequential()                    # (N,3)
model.add(Dense(10,input_shape=(3,)))# (batch_size,input_dim)
model.add(Dense(units=15))   # 아웃풋 노드의 갯수    #출력 (batch_size, units)
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                40
_________________________________________________________________
dense_1 (Dense)              (None, 15)                165
=================================================================
Total params: 205
Trainable params: 205
Non-trainable params: 0
_________________________________________________________________
'''

