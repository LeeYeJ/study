from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import accuracy_score


#실습

#1.데이터
# reshape 해주자~ cnn 성능보다 좋게 만들어라~
(x_train,y_train),(x_test,y_test) =fashion_mnist.load_data()
print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,) (10000,)
print()

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

x_train= x_train.reshape(60000,28*28)
x_test= x_test.reshape(10000,28*28)
print(x_train.shape, x_test.shape) #(60000, 784) (10000, 784)

#2.모델 구성
model= Sequential()
model.add(Dense(8,input_shape=(784,))) # 28*28로 표현해줘도됨
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(9,))
model.add(Dense(7,))
model.add(Dense(10,activation='softmax'))

# 소프트맥스는 라벨값들의 확률이 나오기 때문에 마지막에 알그맥스로 변환해준다.
import time
start_time=time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam')

import datetime # 시간을 저장해줌
date = datetime.datetime.now() # 현재 시간
print(date) # 2023-03-14 11:15:39.585470
date = date.strftime('%m%d_%H%M') # 시간을 문자로 바꾼다 ( 월, 일, 시 ,분)
print(date) # 0314_1115

filepath='./_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #val_loss:4f 소수 넷째자리까지 받아와라

es=EarlyStopping(monitor='loss',mode='auto',patience=10)

mcp= ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # val_loss 기준, verbose=1 훈련중 확인 가능
                    save_best_only=True,  # 가장 좋은 지점에서 세이브하기
                    filepath="".join([filepath, 'k27_', date,'_',filename ])) # 경로는 이곳에 / .join 합친다는 뜻


model.fit(x_train,y_train,epochs=100, batch_size=512,validation_split=0.1, callbacks=[es,mcp])
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

'''
Epoch 00100: val_loss did not improve from 0.99393
313/313 [==============================] - 1s 3ms/step - loss: 1.0098
results : 1.009791374206543
acc : 0.6417
time: 80.21

DNN일떄
Epoch 00050: val_loss did not improve from 1.65160
313/313 [==============================] - 1s 2ms/step - loss: 1.6398
results : 1.6398323774337769
acc : 0.4023
time: 36.23
'''
