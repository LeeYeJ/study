# 중간중간 저장할수있음
# 모델을 저장할 때 사용되는 콜백함수입니다.

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model # Model 인풋 레이어를 따로 명시 / Sequential는 히든레이어와 같이 명시
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

#훈련데이터만 정규화한다.

#2.모델 구성 (Sequential)
# model=Sequential() # 모델 정의 제일 위에서
# model.add(Dense(30, input_shape=(13,))) # 열의 갯수만 땡겨온다 (벡터형식으로 나타내어준것)
# model.add(Dense(20)) # 히든 30,20,10
# model.add(Dense(10))
# model.add(Dense(1))

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

# Model의 weight 값을 중간 저장해 줍니다
mcp= ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # val_loss 기준, verbose=1 훈련중 확인 가능
                    save_best_only=True,  # 가장 좋은 지점에서 세이브하기           
# save_best_only=True 설명 -> True 인 경우, monitor 되고 있는 값을 기준으로 가장 좋은 값으로 모델이 저장됩니다.
# False인 경우, 매 에폭마다 모델이 filepath{epoch}으로 저장됩니다. (model0, model1, model2....)

                    filepath='./_save/MCP/keras27_ModelCheckPoint1.hdf5') # 경로는 이곳에
# shift + tab -> 왼쪽으로 땡기기 (탭의 반대 방향)

model.fit(x_train,y_train,epochs=10000,
          callbacks=[es, mcp], validation_split=0.1)


loss=model.evaluate(x_test,y_test)



print('loss :',loss)




