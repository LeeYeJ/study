#전처리 스케일러

#핏에서 validation_split=0.2 써줌

#모델링까지
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Flatten,Conv2D,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

#데이터 불러오기
path = './_data/kaggle_bike/'
path_save='./_save/kaggle_bike/'
train_csv= pd.read_csv(path +'train.csv', index_col=0)
test_csv=pd.read_csv(path + 'test.csv', index_col=0)

print(train_csv)
print(train_csv.shape) # (10886, 11)

print(test_csv)
print(test_csv.shape) # (6493, 8)

#데이터 결측치 제거
print(train_csv.isnull().sum()) # 결측값 없음

#데이터 x,y분리
x=train_csv.drop(['count','casual','registered'], axis=1)
print(x.columns)
'''
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed'], 컬럼 8개 'casual','registered'는 테스트 데이터에 없으니까 그냥 일단 삭제해보자
'''
y=train_csv['count']
print(y)

# 데이터 split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=7994,train_size=0.9
)
print(x_train.shape,x_test.shape) # (9797, 8) (1089, 8)
print(y_train.shape,y_test.shape) # (9797,) (1089,)

scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
# scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

#test 파일도 스케일링 해줘야됨!!!!!!!!!
test_csv=scaler.transform(test_csv)

# 모델 구성
# model=Sequential()
# model.add(Dense(5,input_dim=8))
# model.add(Dense(6))
# model.add(Dense(6,activation='relu')) # 음수값 조정할때 활성화 함수나 한정화 함수로 각 층에서 던져줄때 조정해줄수있음 
# model.add(Dense(5,activation='relu')) # (예를들면 음수에서 양수화해서 던져줌)
# model.add(Dense(6,activation='relu'))
# model.add(Dense(6))
# model.add(Dense(6))
# model.add(Dense(1))

x_train= x_train.reshape(9797,8,1)
x_test= x_test.reshape(1089,8,1)
test_csv = test_csv.reshape(6493,8,1) # test파일도 모델에서 돌려주니까 리쉐잎 해줘야됨.

model = Sequential()
model.add(LSTM(16,input_shape=(8,1),activation='linear'))
model.add(Dense(9,activation='relu'))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(1))
# 함수형 모델
# input1 = Input(shape=(8,))
# modell = Dense(5,input_dim=8)(input1)
# model2 = Dense(6)(modell)
# model3 = Dense(6,activation='relu')(model2)
# model4 = Dense(5,activation='relu')(model3)
# model5 = Dense(6,activation='relu')(model4)
# model6 = Dense(6)(model5)
# model7 = Dense(6)(model6)
# output1 = Dense(1)(model7)
# model= Model(inputs=input1,outputs=output1)

es=EarlyStopping(mode='min',monitor='val_loss',patience=20,restore_best_weights=True)

#컴파일 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=500,batch_size=200,validation_split=0.2,callbacks=[es])

#평가
loss= model.evaluate(x_test,y_test)
print('loss :',loss)

#예측 r2스코어 확인
y_pre=model.predict(x_test)
r2=r2_score(y_pre,y_test)
print('r2 스코어 :', r2)

# RMSE 함수 정의
def RMSE(y_test,y_pre):
    return np.sqrt(mean_squared_error(y_test,y_pre)) #정의
rmse=RMSE(y_test,y_pre) #사용
print('RMSE :',rmse)

#카운트값 빼기
y_submit = model.predict(test_csv)
# print(y_submit)

#카운트값 넣어주기
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)
# print(submission)

#제출파일의 count에 y_submit을 넣어준다.
submission['count'] = y_submit
# print(submission)

submission.to_csv(path_save + 'submit_0308_1941.csv')

'''
CNN모델
Epoch 250/500
40/40 [==============================] - 0s 2ms/step - loss: 108.8125 - val_loss: 111.5558
35/35 [==============================] - 0s 647us/step - loss: 109.0051
loss : 109.00511169433594
r2 스코어 : -a1.3804669423203264
RMSE : 157.06389618689056

rnn모델
loss : 107.31344604492188
r2 스코어 : -1.5214198586039887
RMSE : 156.44168510431462

'''












