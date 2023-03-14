#사이킷런 로드와인
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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

# scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

model=Sequential()
model.add(Dense(10,activation='relu',input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10,activation='relu'))
model.add(Dense(10))
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
MinMaxScaler

Epoch 59/500
144/144 [==============================] - 0s 897us/step - loss: 0.0055 - acc: 1.0000 - val_loss: 0.0649 - val_acc: 0.9375
1/1 [==============================] - 0s 99ms/step - loss: 0.0224 - acc: 1.0000
loss : [0.022414786741137505, 1.0]
acc : 1.0

StandardScaler

Epoch 500/500
144/144 [==============================] - 0s 865us/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 2.9057e-06 - val_acc: 1.0000
1/1 [==============================] - 0s 114ms/step - loss: 9.5967e-05 - acc: 1.0000
loss : [9.596664312994108e-05, 1.0]
acc : 1.0

RobustScaler

Epoch 28/500
144/144 [==============================] - 0s 931us/step - loss: 7.0532e-04 - acc: 1.0000 - val_loss: 0.3604 - val_acc: 0.9375
1/1 [==============================] - 0s 114ms/step - loss: 0.1688 - acc: 0.9444
loss : [0.16883224248886108, 0.9444444179534912]
acc : 0.9444444444444444

MaxAbsScaler

Epoch 67/500
144/144 [==============================] - 0s 949us/step - loss: 0.0510 - acc: 0.9792 - val_loss: 0.1249 - val_acc: 0.9375
1/1 [==============================] - 0s 109ms/step - loss: 0.0782 - acc: 0.9444
loss : [0.07818803191184998, 0.9444444179534912]
acc : 0.9444444444444444

'''
'''
MinMaxScaler 


StandardScaler 


RobustScaler 


MaxAbsScaler
'''


