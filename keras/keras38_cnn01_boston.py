'''
과적합 배제 방법
1. 데이터가 많아야됨
2. 노드의 일부를 빼면 됨

model.evaluate에선 dropout 안되고 원래 노드만큼 평가된다 그래야 과적합이 안됨.
'''
#어떻게 성능 향상할거야? 일부 노드 뺄게

# 저장할때 지표(평가결과)값, 훈련 시간 등을 파일에 넣어줘

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model # Model 인풋 레이어를 따로 명시 / Sequential는 히든레이어와 같이 명시
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D,Flatten
import numpy as np
from sklearn.preprocessing import MinMaxScaler , StandardScaler 
from sklearn.preprocessing import MaxAbsScaler,RobustScaler


#1.데이터
datasets= load_boston()
x=datasets.data
y=datasets['target']

print(type(x)) #<class 'numpy.ndarray'>
print(x.shape) # (506, 13)


x_train,x_test,y_train,y_test=train_test_split(
    x,y,train_size=0.8,random_state=333
)

print(x_train.shape,x_test.shape) #(404, 13) (102, 13)
# 전처리는 데이터 분리 다음에 해준다.
# print(np.min(x),np.max(x)) #0.0 711.0 (0~711까지)
scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
# scaler= MaxAbsScaler()

# scaler.fit(x_train) # fit의 범위가 x_train이다
# x_train=scaler.transform(x_train) #변환시키라
x_train = scaler.fit_transform(x_train) #위에 두줄을 한줄로 쓸수있다

x_test=scaler.transform(x_test) # x_train의 범위에 맞춰서 변환해준다. 그래서 fit은 할 필요 없음
# print(np.min(x_test),np.max(x_test)) 


print(x_train.shape) # (404, 13)
print(x_test.shape) #  (102, 13)

x_train= x_train.reshape(404,13,1,1)
x_test= x_test.reshape(102,13,1,1)


model = Sequential()
model.add(Conv2D(7,(2,1),input_shape=(13,1,1)))
model.add(Conv2D(8,(2,1),activation='relu'))
model.add(Conv2D(5,(2,1),padding='same'))
model.add(Flatten())
model.add(Dense(9,activation='relu'))
model.add(Dense(6))
model.add(Dense(1))

model.summary()

# 컴파일 훈련
model.compile(loss='mse',optimizer='adam')

import datetime # 시간을 저장해줌
date = datetime.datetime.now() # 현재 시간
print(date) # 2023-03-14 11:15:39.585470
date = date.strftime('%m%d_%H%M') # 시간을 문자로 바꾼다 ( 월, 일, 시 ,분)
print(date) # 0314_1115

filepath='./_save/MCP/keras28/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #val_loss:4f 소수 넷째자리까지 받아와라


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint 
es = EarlyStopping(monitor='val_loss', patience=10,mode='min',
                   verbose=1,
                   restore_best_weights=True
                   )
mcp= ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # val_loss 기준, verbose=1 훈련중 확인 가능
                    save_best_only=True,  # 가장 좋은 지점에서 세이브하기
                    filepath="".join([filepath, 'k28_1', date,'_',filename ])) # 경로는 이곳에 / .join 합친다는 뜻

model.fit(x_train,y_train,epochs=10000,
          callbacks=[es,mcp], # mcp 
          validation_split=0.1)


# filepath='./_save/MCP/keras27_ModelCheckPoint1.hdf5') 경로로 잡아줌
# model = load_model('./_save/MCP/keras27_ModelCheckPoint1.hdf5') # 로스 고정

# 평가 예측
from sklearn.metrics import r2_score

print('==================1. 기본출력===============')
loss=model.evaluate(x_test,y_test, verbose=0)
print('loss :',loss)

y_predict = model.predict(x_test)
r2= r2_score(y_predict,y_test)
print('r2 :',r2)

'''
CNN 해본 결과
loss : 19.363056182861328
r2 : 0.7350176617468278

'''


