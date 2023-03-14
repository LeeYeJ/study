from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model # Model 인풋 레이어를 따로 명시 / Sequential는 히든레이어와 같이 명시
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler , StandardScaler 
from sklearn.preprocessing import MaxAbsScaler,RobustScaler


#1.데이터
datasets= load_boston()
x=datasets.data
y=datasets['target']

print(type(x)) #<class 'numpy.ndarray'>
print(x)


x_test,x_train,y_test,y_train=train_test_split(
    x,y,train_size=0.8,random_state=333
)
# 전처리는 데이터 분리 다음에 해준다.
print(np.min(x),np.max(x)) #0.0 711.0 (0~711까지)
scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝

# scaler.fit(x_train) # fit의 범위가 x_train이다
# x_train=scaler.transform(x_train) #변환시키라
x_train = scaler.fit_transform(x_train) #위에 두줄을 한줄로 쓸수있다

x_test=scaler.transform(x_test) # x_train의 범위에 맞춰서 변환해준다. 그래서 fit은 할 필요 없음
print(np.min(x_test),np.max(x_test)) 
'''


# 함수형 모델
input1 = Input(shape=(13,)) # 스칼라 13개 벡터 1개
dense1 = Dense(30)(input1) #(input1) 이 레이어가 어디서 왔는지
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1,outputs=output1) # 시작과 끝을 명시 / 모델 정의 제일 아래서

# model.save('./_save/keras26_1_save_model.h5')

model.compile(loss='mse',optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint 
es = EarlyStopping(monitor='val_loss', patience=10,mode='min',
                   verbose=1,restore_best_weights=True)
mcp= ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # val_loss 기준, verbose=1 훈련중 확인 가능
                    save_best_only=True,  # 가장 좋은 지점에서 세이브하기
                    filepath='./_save/MCP/keras27_ModelCheckPoint1.hdf5') # 경로는 이곳에
# shift + tab -> 왼쪽으로 땡기기 (탭의 반대 방향)

model.fit(x_train,y_train,epochs=10000,
          callbacks=[es, mcp], validation_split=0.1)

'''
# filepath='./_save/MCP/keras27_ModelCheckPoint1.hdf5') 경로로 잡아줌
model = load_model('./_save/MCP/keras27_ModelCheckPoint1.hdf5') # 로스 고정


loss=model.evaluate(x_test,y_test)
print('loss :',loss)




