import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test) =mnist.load_data()
print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,) (10000,)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape) #(60000, 10) (10000, 10)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

x_train= x_train.reshape(60000,28,28,1)
x_test= x_test.reshape(10000,28,28,1)


model = Sequential()
model.add(Dense(64,input_shape=(28,28,)))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(10,activation='softmax'))

model.summary()

# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train, epochs=100, batch_size=128,)

results= model.evaluate(x_test,y_test)
print('loss:', results[0])
print('acc:',results[1])
