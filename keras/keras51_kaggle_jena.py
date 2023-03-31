# 시계열로 LSTM
# 판다스로 데이터를 불러오고 넘파이로 바꿔서 연산시킨다.
# RMSE로 하자 loss에 mse /metrics에 mae / 7:2: 1 (predict로 rmse빼라)

# ctrl + F -> 같은 네임 바꿔줄수있음
# ctrl + F2 -> 같은 네임 한번에 바꿀수있음

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Flatten,Conv2D,LSTM,Conv1D,Reshape
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# print(datasets) # datetime은 연산 안된 인덱스 취급하자
# print(datasets.shape) #(420551, 14)

x = datasets.drop(['T (degC)'],axis=1)
y = datasets['T (degC)']
# print(x.shape) # (420551, 13)
# print(y.shape) # (420551,)

# print(datasets.columns)
# print(datasets.info()) # 결측없음
# print(datasets.describe()) # 온도가 y값

x_train,x_test,y_train,y_test=train_test_split(
    x, y, shuffle=False, train_size=0.7 
)
# print(x_train.shape)
# print(x_test.shape)

x_test,x_predict,y_test,y_predict = train_test_split(
    x_test,y_test, shuffle=False, train_size=0.67 # train_size에 의미부여 하지말고 앞에 데이터에 비율이 들어감
)
# print(x_train.shape) # (294385, 13)
# print(x_test.shape) # (84531, 13)
# print(x_predict.shape) # (41635, 13)

# x_train = np.array(x_train).reshape(294385,13,1)/255.
# x_test = np.array(x_test).reshape(84531,13,1)/255.
# x_predict = np.array(x_predict).reshape(41635,13,1)/255.

# print(x_train.shape, x_test.shape)

timesteps=10

# print(x_train.shape)

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps ):
        subset = dataset[i:(i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x_train = split_x(x_train,timesteps)


x_test = split_x(x_test,timesteps)

x_predict = split_x(x_predict,timesteps)

# print(x_train.shape) # (294376, 10, 13)


y_train = y_train[timesteps:]
y_test = y_test[timesteps:]
y_predict = y_predict[timesteps:]
print(y_train.shape)


model = Sequential()
model.add(LSTM(16,input_shape=(10,13),activation='linear')) # 행 빼고 나머지/ 다른 모델들도 마찬가지임 / 32는 아웃풋 노드의 갯수
model.add(Dense(16,activation='relu'))
model.add(Dense(8))
model.add(Reshape(target_shape=(4,2)))
model.add(LSTM(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(10,activation='relu'))
model.add(Dense(16))
model.add(Dense(1))

model.summary()

# model.load_weights('./_save/kaggle_jena_save_weights1.h5')

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
import time
start_time = time.time()
es=EarlyStopping(monitor='loss',mode='auto',patience=20,restore_best_weights=True)
model.fit(x_train,y_train,epochs=500,batch_size=500,callbacks=[es])
end_time= time.time()

model.save_weights('./_save/kaggle_jena_save_weights2.h5')

#4.평가예측
loss = model.evaluate(x_test,y_test)

predict = model.predict(x_predict)

r2 = r2_score(predict,y_predict)
print("r2 스코어 : ", r2)

def RMSE(y_test,y_predict): # 함수를 정의할때 사용 ():안에 입력값을 받아서 
    return np.sqrt(mean_squared_error(predict,y_predict)) # RMSE 함수 정의
rmse = RMSE(predict, y_predict)                           # RMSE 함수 사용
print("RMSE : ", rmse)
# result = model.predict()
print('loss : ', loss)
# print('[100:107]의 결과값 :', result)
print('걸린 시간은 :', round(end_time-start_time,2))



# print(datasets['T (degC)'])
# print(datasets['T (degC)'].values) #넘파이 형태로 바꿔준것. #[-8.02 -8.41 -8.51 ... -3.16 -4.23 -4.82]
# print(datasets['T (degC)'].to_numpy()) # 얘 또한 마찬가지 #[-8.02 -8.41 -8.51 ... -3.16 -4.23 -4.82]


# import matplotlib.pyplot as plt # 넘파이 형태의 데이터여야 한다.
# plt.plot(datasets['T (degC)','y_predict'].values) # 판다스 형식이 아닌 넘파이 형식으로 바꿔줘야 됨.
# plt.show()

'''
2642/2642 [==============================] - 3s 1ms/step - loss: 1132.3329
r2 스코어 :  0.9980574222456099
RMSE :  0.3438116255089698
loss :  1132.3328857421875
걸린 시간은 : 76.09

2600/2642 [============================>.] - ETA: 0s - loss2642/2642 [==============================] - 3s 1ms/step - 
loss: 222.1315
r2 스코어 :  0.9992452732214017
RMSE :  0.21558813308090646
loss :  222.1314697265625
걸린 시간은 : 1851.62

LSTM 두번 썼더니 성능 확 좋아짐
Epoch 142/500
589/589 [==============================] - 5s 8ms/step - loss: 0.0532
2642/2642 [==============================] - 4s 1ms/step - loss: 0.1952
r2 스코어 :  0.9991062789807046
RMSE :  0.23346033801214472
loss :  0.19517961144447327
걸린 시간은 : 675.26
'''
