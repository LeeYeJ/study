# test 파일도 스케일링 해줘야됨!!!!!!!!! (train도 해줬으니까)

import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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

#데이터분리

x_train,x_test,y_train,y_test=train_test_split(
 x,y,shuffle=True,random_state=4897567,test_size=0.1
)
# scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test) 

#test 파일도 스케일링 해줘야됨!!!!!!!!!
test_csv=scaler.transform(test_csv)

#모델
model = Sequential()
model.add(Dense(6, input_dim = 9))
model.add(Dense(8))
model.add(Dense(9,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(6))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

#컴파일
es=EarlyStopping(monitor='val_loss',mode='min',patience=100,restore_best_weights=True)

model.compile(loss='mae',optimizer='adam')
hist=model.fit(x_train,y_train,epochs=2000, batch_size=150,validation_split=0.1,callbacks=[es])

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
MinMaxScaler

Epoch 790/2000
215/215 [==============================] - 0s 857us/step - loss: 28.9987 - val_loss: 37.1766
Epoch 791/2000
215/215 [==============================] - 0s 859us/step - loss: 29.6335 - val_loss: 35.6931
5/5 [==============================] - 0s 0s/step - loss: 38.1767
loss: 38.176734924316406
r2: 0.5967476788466672
rmse: 56.34336557020495

StandardScaler

Epoch 1330/2000
8/8 [==============================] - 0s 3ms/step - loss: 34.9277 - val_loss: 40.8313
5/5 [==============================] - 0s 801us/step - loss: 42.2698
loss: 42.26976013183594
r2: 0.5267384546098527
rmse: 61.03865510795263

RobustScaler

Epoch 258/2000
8/8 [==============================] - 0s 4ms/step - loss: 37.3861 - val_loss: 43.2293
5/5 [==============================] - 0s 726us/step - loss: 42.7622
loss: 42.76216506958008
r2: 0.5185229952791017
rmse: 61.56616789310163

MaxAbsScaler

Epoch 335/2000
8/8 [==============================] - 0s 4ms/step - loss: 36.6257 - val_loss: 43.1831
5/5 [==============================] - 0s 759us/step - loss: 43.1477
loss: 43.147705078125
r2: 0.4957892199748889
rmse: 63.00288118130614

'''

