from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Input ,LSTM,Conv1D,Flatten
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


scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
# scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)


# print(x_train.shape) # (397, 10)
# print(x_test.shape) # (45, 10)

# model=Sequential()
# model.add(Dense(7, input_dim=10))
# model.add(Dense(8))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(8))
# model.add(Dense(1))

x_train= x_train.reshape(397,5,2)
x_test= x_test.reshape(45,5,2)

model = Sequential()
model.add(Conv1D(16,2,input_shape=(5,2),activation='linear'))
model.add(Conv1D(10,3))
model.add(Flatten())
model.add(Dense(9,activation='relu'))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(1))

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
cnn모델
Epoch 115/1000
18/18 [==============================] - 0s 7ms/step - loss: 45.4208 - val_loss: 52.7556
2/2 [==============================] - 0s 122ms/step - loss: 37.8659
loss : 37.86589431762695
r2: 0.3414550953909341

Epoch 114/1000
18/18 [==============================] - 0s 7ms/step - loss: 45.3819 - val_loss: 53.1728
2/2 [==============================] - 0s 128ms/step - loss: 37.3734
loss : 37.373443603515625
r2: 0.45175072528312954

rnn 해본 모델
loss : 49.550411224365234
r2: -0.43933145400724505

Conv1 해본 결과
oss : 35.4218635559082
r2: 0.43666548653807435
'''