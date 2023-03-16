import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler # 2차원에서만 됨
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

(x_train,y_train),(x_test,y_test) =cifar10.load_data()

print(x_train.shape,y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape) # (10000, 32, 32, 3) (10000, 1)

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)  

print(x_train.shape) 
print(x_test.shape)  

x_train = x_train/255.
x_test = x_test/255.

# x_train = x_train.reshape(50000,32*32*3) # 구조만 달라지고 순서와 값은 바뀌지 않는다.
# x_test = x_test.reshape(10000,32*32*3)

# print(np.unique(y_train,return_counts=True)) #(array([0., 1.], dtype=float32), array([450000,  50000], dtype=int64)
 
# # print(y.shape)

# Scaler= MinMaxScaler()
# Scaler.fit(x_train)
# x_train = Scaler.transform(x_train)
# x_test = Scaler.transform(x_test)

# x_train = x_train.reshape(50000,32,32,3) # 구조만 달라지고 순서와 값은 바뀌지 않는다.
# x_test = x_test.reshape(10000,32,32,3)

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(32,32,3)))
model.add(MaxPooling2D())
model.add(Conv2D(10,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(5,(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(8,activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

# 소프트맥스는 라벨값들의 확률이 나오기 때문에 마지막에 알그맥스로 변환해준다.
import time
start_time=time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam')
es=EarlyStopping(monitor='loss',mode='auto',patience=10)
model.fit(x_train,y_train,epochs=50, batch_size=512,validation_split=0.1, callbacks=[es])
end_time = time.time()

results= model.evaluate(x_test,y_test)
print('results :', results)

y_pred=model.predict(x_test)
y_test_acc=np.argmax(y_test,axis=1) 
# print(y_test_acc) 
y_pred=np.argmax(y_pred,axis=1)
# print(y_pre) 

acc=accuracy_score(y_pred,y_test_acc)   
print('acc :', acc)

print('time:', round(end_time-start_time,2))

