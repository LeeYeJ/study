import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential, Model, Input
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler # 2차원에서만 됨
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

(x_train,y_train),(x_test,y_test) =cifar10.load_data()

print(x_train.shape,y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape) # (10000, 32, 32, 3) (10000, 1)

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)  

print(y_train.shape) 
print(y_test.shape)  

x_train = x_train/255.
x_test = x_test/255.

print(x_train.shape) 
print(x_test.shape) 

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


# 함수형 모델
input1 = Input(shape=(32,32,3)) 
conv1 = Conv2D(30,(2,2))(input1)
max1 = MaxPooling2D()(conv1)
conv2= Conv2D(filters=10,kernel_size=(3,3),activation='relu')(max1)
max2 = MaxPooling2D()(conv2)
conv3 = Conv2D(5,2,padding='same')(max2)
flat1= Flatten()(conv3)
dense1 = Dense(8,activation='relu')(flat1) 
dense2 = Dense(8,activation='relu')(dense1)
drop1 = Dropout(0.2)(dense2)
output1 = Dense(10,activation='softmax')(drop1)
model = Model(inputs = input1,outputs=output1) 

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
Epoch 50/50
88/88 [==============================] - 12s 137ms/step - loss: 1.2666 - val_loss: 1.2372
313/313 [==============================] - 1s 3ms/step - loss: 1.2406
results : 1.2405617237091064
acc : 0.5574
time: 556.5
'''