#사이킷런 로드와인
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D,Flatten,LSTM
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

datasets=load_wine()
x=datasets.data
y=datasets.target

print(y)

print(x.shape,y.shape) # (178, 13) (178,)

print(np.unique(y)) # [0 1 2] # y라벨 추출

#######다중이니까 원핫인코딩해주기###########

y = to_categorical(y)  # 0부터 시작
print(y)
print(y.shape) #(178, 3)

##########################################

x_train,x_test,y_train,y_test=train_test_split(
    x,y, shuffle=True, random_state=3338478, train_size=0.9, stratify=y
)

print(x_train.shape) #(160, 13)
print(x_test.shape) #(18, 13)

# scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

x_train= x_train.reshape(160,13,1)
x_test= x_test.reshape(18,13,1)

model = Sequential()
model.add(LSTM(16,input_shape=(13,1),activation='linear'))
model.add(Dense(9,activation='relu'))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(3,activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

es=EarlyStopping(monitor='val_loss',mode='auto',patience=20,restore_best_weights=True)

model.fit(x_train,y_train,epochs=500,batch_size=1,validation_split=0.1,callbacks=[es])

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_pre=np.argmax(model.predict(x_test),axis=1)


y_pre = to_categorical(y_pre)
# print(y_pre)
# print(y_pre.shape) #(178, 3)


y_pre=np.round(model.predict(x_test))
# print(y_pre)
# print("============")
# print(y_test)

acc=accuracy_score(y_test,y_pre)
print('acc :', acc)

'''
CNN모델
Epoch 186/500
144/144 [==============================] - 0s 986us/step - loss: 5.7949e-09 - acc: 1.0000 - val_loss: 0.0098 - val_acc: 1.0000
1/1 [==============================] - 0s 111ms/step - loss: 0.2368 - acc: 0.8889
loss : [0.23677243292331696, 0.8888888955116272]
acc : 0.8888888888888888

rnn모델
1/1 [==============================] - 0s 273ms/step - loss: 0.3839 - acc: 0.9444
loss : [0.3838687837123871, 0.9444444179534912]
acc : 0.9444444444444444
'''


