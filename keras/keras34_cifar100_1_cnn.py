import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler # 2차원에서만 됨
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

(x_train,y_train),(x_test,y_test) = cifar100.load_data()

print(x_train.shape,y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape) # (10000, 32, 32, 3) (10000, 1)

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)  

print(x_train.shape) 
print(x_test.shape)  

x_train = x_train/255.
x_test = x_test/255.

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(32,32,3)))
model.add(MaxPooling2D())
model.add(Conv2D(20,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(5,(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(8,activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='softmax')) # = units의 갯수가 100개 

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


model.fit(x_train,y_train,epochs=50, batch_size=512,validation_split=0.1, callbacks=[es,mcp])
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
Epoch 00050: val_loss improved from 3.51315 to 3.49914, saving model to ./_save/MCP/keras27_4\k27_0317_1008_0050-3.499141.hdf5
313/313 [==============================] - 1s 3ms/step - loss: 3.4612
results : 3.461217164993286
acc : 0.165
time: 536.78
'''
