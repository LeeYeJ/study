import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , LSTM, Dropout,Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

#데이터
path = './_data/tesla_test/'

dataset = pd.read_csv(path+'TSLA.csv',index_col=0)

print(dataset.isna().info()) # (251, 6)
print(dataset.shape)

x = dataset.drop(['Close'],axis=1)
y = dataset['Close']
print(x.shape,y.shape) #(251, 5) (251,)

x_train,x_test,y_trian,y_test=train_test_split(
    x, y, shuffle=False, train_size=0.7
)
print(x_train.shape,x_test.shape) # 

x_test,x_pred,y_test,y_pred = train_test_split(
    x_test,y_test, shuffle=False, train_size=0.67
)

def split_x(dataset, timesteps):
    a = []
    for i in range(len(dataset)-timesteps):
        sub = dataset[i:(i+timesteps)]
        a.append(sub)
    return np.array(a)

x_train = split_x(x_train,3)
x_test = split_x(x_test,3)
x_pred = split_x(x_pred,3)

print(x_train.shape,x_test.shape,x_pred.shape) #(172, 3, 5) (47, 3, 5) (23, 3, 5)

y_train = y_train[timesteps:]
y_test = y_test[timesteps:]
y_predict = y_predict[timesteps:]


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








