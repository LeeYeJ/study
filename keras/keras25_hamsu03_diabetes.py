from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=650874,train_size=0.9
)
##(442, 10) (442,)

scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
# scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

# model=Sequential()
# model.add(Dense(7, input_dim=10))
# model.add(Dense(8))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(8))
# model.add(Dense(1))

input1 = Input(shape=(10,))
modell = Dense(7)(input1)
model2 = Dense(8)(modell)
model3 = Dense(8,activation='relu')(model2)
model4 = Dense(8,activation='relu')(model3)
model5 = Dense(8)(model4)
output1 = Dense(1)(model5)
model= Model(inputs=input1,outputs=output1)

# es=EarlyStopping(monitor='val_loss',patience=100,mode='min',restore_best_weights=True)

es=EarlyStopping(monitor='val_loss', patience=20,mode='min',restore_best_weights=True)

model.compile(loss='mae',optimizer='adam')
hist = model.fit(x_train,y_train,epochs=1000,batch_size=20,validation_split=0.1,callbacks=[es])

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_pre=model.predict(x_test)
r2=r2_score(y_pre,y_test)
print('r2:', r2)

plt.plot(hist.history['val_loss'])
plt.plot(hist.history['loss'])

plt.show()

# 얼리 스타핑은 로스값을 계속 비교해주다가 patience 횟수까지 더 나아지지 않으면 정지

'''
MinMaxScaler

Epoch 72/1000
18/18 [==============================] - 0s 2ms/step - loss: 44.2180 - val_loss: 54.0784
Epoch 73/1000
18/18 [==============================] - 0s 2ms/step - loss: 44.1741 - val_loss: 53.9546
2/2 [==============================] - 0s 16ms/step - loss: 32.6264
loss : 32.626365661621094
r2: 0.5497248825630285

StandardScaler

Epoch 57/1000
18/18 [==============================] - 0s 1ms/step - loss: 43.8452 - val_loss: 53.9775
Epoch 58/1000
18/18 [==============================] - 0s 2ms/step - loss: 43.7885 - val_loss: 54.2163
2/2 [==============================] - 0s 0s/step - loss: 33.8931
loss : 33.893131256103516
r2: 0.5027127628833633

RobustScaler

Epoch 64/1000
18/18 [==============================] - 0s 2ms/step - loss: 44.0578 - val_loss: 53.7650
Epoch 65/1000
18/18 [==============================] - 0s 2ms/step - loss: 44.0758 - val_loss: 53.7134
2/2 [==============================] - 0s 1ms/step - loss: 34.1961
loss : 34.19609069824219
r2: 0.5071664262499591

MaxAbsScaler

Epoch 60/1000
18/18 [==============================] - 0s 2ms/step - loss: 43.9294 - val_loss: 53.6658
Epoch 61/1000
18/18 [==============================] - 0s 2ms/step - loss: 43.9610 - val_loss: 53.6569
2/2 [==============================] - 0s 1ms/step - loss: 34.9953
loss : 34.99531173706055
r2: 0.5090285100759029

'''
'''
함수형 모델

MinMaxScaler 

Epoch 74/1000
18/18 [==============================] - 0s 2ms/step - loss: 44.5282 - val_loss: 53.9175
2/2 [==============================] - 0s 969us/step - loss: 35.8258
loss : 35.82581329345703
r2: 0.45973308959752524

StandardScaler 

Epoch 57/1000
18/18 [==============================] - 0s 2ms/step - loss: 42.3675 - val_loss: 54.3319
Epoch 58/1000
18/18 [==============================] - 0s 2ms/step - loss: 42.3907 - val_loss: 53.8221
2/2 [==============================] - 0s 1ms/step - loss: 32.5306
loss : 32.53055191040039
r2: 0.6967104467578189

RobustScaler 

Epoch 56/1000
18/18 [==============================] - 0s 2ms/step - loss: 43.8312 - val_loss: 53.4356
2/2 [==============================] - 0s 0s/step - loss: 34.0352
loss : 34.03520584106445
r2: 0.6569419223307418

MaxAbsScaler

Epoch 110/1000
18/18 [==============================] - 0s 2ms/step - loss: 42.4347 - val_loss: 52.5318
2/2 [==============================] - 0s 0s/step - loss: 31.7853
loss : 31.785329818725586
r2: 0.691385140756212

'''