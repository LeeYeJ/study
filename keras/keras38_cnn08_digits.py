#사이킷런 로드 디짓


#사이킷런 로드와인
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

datasets=load_digits()
x=datasets.data
y=datasets.target

print(y)

print(x.shape,y.shape) # (1797, 64) (1797,)

print(np.unique(y)) # [0 1 2 3 4 5 6 7 8 9] # y라벨 추출

#######다중이니까 원핫인코딩해주기###########

y = to_categorical(y)
print(y) 
print(y.shape) #(1797, 10)

##########################################

x_train,x_test,y_train,y_test=train_test_split(
    x,y, shuffle=True, random_state=3338478, train_size=0.9, stratify=y
)

print(x_train.shape) #(1617, 64)
print(x_test.shape) #(180, 64)

# scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

x_train= x_train.reshape(1617,8,8,1)
x_test= x_test.reshape(180,8,8,1)

model = Sequential()
model.add(Conv2D(7,(2,1),input_shape=(8,8,1)))
model.add(Conv2D(8,(2,1),activation='relu',padding='same'))
model.add(Conv2D(5,(2,1),padding='same'))
model.add(Flatten())
model.add(Dense(9,activation='relu'))
model.add(Dense(6))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

es=EarlyStopping(monitor='val_loss',mode='auto',patience=20,restore_best_weights=True)

model.fit(x_train,y_train,epochs=500,batch_size=50,validation_split=0.1,callbacks=[es])

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_pre=np.round(model.predict(x_test))
acc=accuracy_score(y_test,y_pre)
print('acc :', acc)


'''
MinMaxScaler

Epoch 84/500
30/30 [==============================] - 0s 2ms/step - loss: 0.0516 - acc: 0.9842 - val_loss: 0.3140 - val_acc: 0.9259
6/6 [==============================] - 0s 997us/step - loss: 0.1790 - acc: 0.9222
loss : [0.17904254794120789, 0.9222221970558167]
acc : 0.9166666666666666

StandardScaler

Epoch 45/500
30/30 [==============================] - 0s 2ms/step - loss: 0.0532 - acc: 0.9897 - val_loss: 0.4818 - val_acc: 0.9198
6/6 [==============================] - 0s 798us/step - loss: 0.2911 - acc: 0.9000
loss : [0.29112929105758667, 0.8999999761581421]
acc : 0.8944444444444445

RobustScaler

Epoch 57/500
30/30 [==============================] - 0s 2ms/step - loss: 0.0504 - acc: 0.9911 - val_loss: 0.4646 - val_acc: 0.8889
6/6 [==============================] - 0s 797us/step - loss: 0.2480 - acc: 0.9333
loss : [0.24800541996955872, 0.9333333373069763]
acc : 0.9277777777777778

MaxAbsScaler

Epoch 98/500
30/30 [==============================] - 0s 2ms/step - loss: 0.0576 - acc: 0.9856 - val_loss: 0.3755 - val_acc: 0.9383
6/6 [==============================] - 0s 858us/step - loss: 0.2469 - acc: 0.9556
loss : [0.24689459800720215, 0.9555555582046509]
acc : 0.95

'''
'''
CNN
Epoch 43/500
30/30 [==============================] - 0s 3ms/step - loss: 0.0839 - acc: 0.9766 - val_loss: 0.4713 - val_acc: 0.9259
6/6 [==============================] - 0s 1ms/step - loss: 0.2148 - acc: 0.9333
loss : [0.21477778255939484, 0.9333333373069763]
acc : 0.9111111111111111
'''


