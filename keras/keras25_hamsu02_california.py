import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

datasets= fetch_california_housing()
x=datasets.data
y=datasets['target']
#(20640, 8) (20640,)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=678,test_size=0.3
)

scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
# scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test) # x_train의 범위에 맞춰서 변환해준다. 그래서 fit은 할 필요 없음

# model=Sequential()
# model.add(Dense(7,input_dim=8))
# model.add(Dense(8))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(6,activation='relu'))
# model.add(Dense(9))
# model.add(Dense(1))

input1 = Input(shape=(8,))
modell = Dense(7)(input1)
model2 = Dense(8)(modell)
model3 = Dense(5,activation='relu')(model2)
model4 = Dense(6,activation='relu')(model3)
model5 = Dense(9)(model4)
output1 = Dense(1)(model5)
model= Model(inputs=input1,outputs=output1)

es= EarlyStopping(monitor='val_loss',patience=20,mode='min',restore_best_weights=True)

model.compile(loss='mae',optimizer='adam')
hist=model.fit(x_train,y_train, epochs=200,batch_size=50,validation_split=0.3,callbacks=[es])

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_pre=model.predict(x_test)
r2=r2_score(y_pre,y_test)
print('r2score :', r2)

plt.plot(hist.history['val_loss'])


plt.show()

'''
MinMaxScaler

Epoch 199/200
203/203 [==============================] - 0s 1ms/step - loss: 0.4372 - val_loss: 0.4261
Epoch 200/200
203/203 [==============================] - 0s 1ms/step - loss: 0.4370 - val_loss: 0.4513
194/194 [==============================] - 0s 542us/step - loss: 0.4379
loss : 0.4379250705242157
r2score : 0.6464002549574546

StandardScaler

Epoch 123/200
203/203 [==============================] - 0s 1ms/step - loss: 0.4158 - val_loss: 0.4039
194/194 [==============================] - 0s 604us/step - loss: 0.3951
loss : 0.3950739800930023
r2score : 0.6349552578787483

RobustScaler

203/203 [==============================] - 0s 1ms/step - loss: 0.3816 - val_loss: 0.3725
Epoch 177/200
203/203 [==============================] - 0s 1ms/step - loss: 0.3807 - val_loss: 0.3759
194/194 [==============================] - 0s 502us/step - loss: 0.3647
loss : 0.36474373936653137
r2score : 0.6925393469988825

MaxAbsScaler

Epoch 32/200
203/203 [==============================] - 0s 1ms/step - loss: 0.8867 - val_loss: 0.8881
Epoch 33/200
203/203 [==============================] - 0s 1ms/step - loss: 0.8869 - val_loss: 0.8876
194/194 [==============================] - 0s 569us/step - loss: 0.8760
loss : 0.875957190990448
r2score : -96786953284531.56

'''
'''
함수형 모델

MinMaxScaler
 
Epoch 136/200
203/203 [==============================] - 0s 1ms/step - loss: 0.4537 - val_loss: 0.4483
194/194 [==============================] - 0s 604us/step - loss: 0.4306
loss : 0.43060848116874695
r2score : 0.55491097423012
 
StandardScaler
 
 Epoch 199/200
203/203 [==============================] - 0s 1ms/step - loss: 0.4033 - val_loss: 0.3901
Epoch 200/200
203/203 [==============================] - 0s 1ms/step - loss: 0.4017 - val_loss: 0.3881
194/194 [==============================] - 0s 580us/step - loss: 0.3794
loss : 0.3794138431549072
r2score : 0.6732136559560193

RobustScaler 

Epoch 200/200
203/203 [==============================] - 0s 1ms/step - loss: 0.3833 - val_loss: 0.3784
194/194 [==============================] - 0s 541us/step - loss: 0.3631
loss : 0.3631284534931183
r2score : 0.7008625846142884

MaxAbsScaler

Epoch 200/200
203/203 [==============================] - 0s 1ms/step - loss: 0.4673 - val_loss: 0.4573
194/194 [==============================] - 0s 714us/step - loss: 0.4510
loss : 0.45097795128822327
r2score : 0.616775967575959

'''