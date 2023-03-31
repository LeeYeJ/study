# test 파일도 스케일링 해줘야됨!!!!!!!!! (train도 해줬으니까)

import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Conv2D, Flatten,LSTM
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

path='./_data/ddarung/'
path_save='./_save/ddarung/'

train_csv=pd.read_csv(path+'train.csv', index_col=0)

print(train_csv.shape) #(1459, 10)

test_csv=pd.read_csv(path +'test.csv', index_col=0)

print(test_csv.shape) #(715, 9)

#결측치 제거
print(train_csv.isnull().sum())
train_csv=train_csv.dropna()
print(train_csv.isnull().sum())

#데이터분리!!!!!!!!!!!!!!! 외워 좀!! 
x=train_csv.drop(['count'],axis=1)
y=train_csv['count']

print(train_csv.shape)
print(test_csv.shape)

#데이터분리

x_train,x_test,y_train,y_test=train_test_split(
 x,y,shuffle=True,random_state=4897567,test_size=0.1
)


scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
# scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

#test 파일도 스케일링 해줘야됨!!!!!!!!!
test_csv=scaler.transform(test_csv)

#모델
# model = Sequential()
# model.add(Dense(6, input_dim = 9))
# model.add(Dense(8))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(7,activation='relu'))
# model.add(Dense(6))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1))

print(x_train.shape) # (1195, 9)
print(x_test.shape) # (133, 9)
print(y_test.shape) # (133,)

x_train= x_train.reshape(1195,3,3)
x_test= x_test.reshape(133,3,3)
test_csv = test_csv.reshape(715,3,3) # test파일도 모델에서 돌려주니까 리쉐잎 해줘야됨.

model = Sequential()
model.add(LSTM(16,input_shape=(3,3),activation='linear'))
model.add(Dense(9,activation='relu'))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(1))


# input1 = Input(shape=(9,))
# modell = Dense(6)(input1)
# model2 = Dense(8)(modell)
# model3 = Dense(9,activation='relu')(model2)
# model4 = Dense(7,activation='relu')(model3)
# model5 = Dense(6)(model4)
# model6 = Dense(10,activation='relu')(model5)
# output1 = Dense(1)(model6)

# model= Model(inputs=input1,outputs=output1)

#컴파일
es=EarlyStopping(monitor='val_loss',mode='min',patience=100,restore_best_weights=True)

model.compile(loss='mae',optimizer='adam')
hist=model.fit(x_train,y_train,epochs=20, batch_size=150,validation_split=0.1,callbacks=[es])

loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_pre=model.predict(x_test)
r2=r2_score(y_test,y_pre)
print('r2:',r2)

def RMSE(y_pre,y_test):
    return np.sqrt(mean_squared_error(y_pre,y_test))
rmse=RMSE(y_pre,y_test)

print('rmse:',rmse)

y_sub=model.predict(test_csv)
submission = pd.read_csv(path + 'submission.csv', index_col = 0)
submission['count'] = y_sub
submission.to_csv(path_save + 'submit_0310_1957.csv')

plt.plot(hist.history['val_loss'])
plt.show()
'''
CNN 모델

Epoch 20/20
8/8 [==============================] - 0s 3ms/step - loss: 75.6116 - val_loss: 81.1850
5/5 [==============================] - 0s 749us/step - loss: 84.8362
loss: 84.83616638183594
r2: -0.5871596354691897
rmse: 111.78014078852303

rnn 모델 

loss: 54.20458984375
r2: 0.3371573489750498
rmse: 72.23697814899236
'''

