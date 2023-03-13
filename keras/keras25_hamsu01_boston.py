# 가중치 저장!!

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

#보스턴 임포트 못하는 사람(1.2부터 임포트 안된다.)
# pip uninstall scikit-lean
# pip install scikit-lean==1.1

x_test,x_train,y_test,y_train=train_test_split(
    x,y,train_size=0.8,random_state=333
)
# 전처리는 데이터 분리 다음에 해준다.
print(np.min(x),np.max(x)) #0.0 711.0 (0~711까지)
scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
scaler.fit(x_train) # fit의 범위가 x_train이다
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test) # x_train의 범위에 맞춰서 변환해준다. 그래서 fit은 할 필요 없음
print(np.min(x_test),np.max(x_test)) 

#훈련데이터만 정규화한다.

# 2.모델 구성 (Sequential)
model=Sequential() # 모델 정의 제일 위에서
model.add(Dense(30, input_shape=(13,),name='S1')) # 열의 갯수만 땡겨온다 (벡터형식으로 나타내어준것)
model.add(Dense(20,name='S2')) # 히든 30,20,10
model.add(Dense(10,name='S3'))
model.add(Dense(1,name='S4'))
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
S1 (Dense)                   (None, 30)                420
_________________________________________________________________
S2 (Dense)                   (None, 20)                620
_________________________________________________________________
S3 (Dense)                   (None, 10)                210
_________________________________________________________________
S4 (Dense)                   (None, 1)                 11
=================================================================
Total params: 1,261
Trainable params: 1,261
Non-trainable params: 0

'''

# 함수형 모델
input1 = Input(shape=(13,),name='H1') # 스칼라 13개 벡터 1개
dense1 = Dense(30,name='H2')(input1) #(input1) 이 레이어가 어디서 왔는지
dense2 = Dense(20,name='H3')(dense1)
dense3 = Dense(10,name='H4')(dense2)
output1 = Dense(1,name='H5')(dense3)
model = Model(inputs = input1,outputs=output1) # 시작과 끝을 명시 / 모델 정의 제일 아래서
model.summary()
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
H1 (InputLayer)              [(None, 13)]              0
_________________________________________________________________
H2 (Dense)                   (None, 30)                420
_________________________________________________________________
H3 (Dense)                   (None, 20)                620
_________________________________________________________________
H4 (Dense)                   (None, 10)                210
_________________________________________________________________
H5 (Dense)                   (None, 1)                 11
=================================================================
Total params: 1,261
Trainable params: 1,261
Non-trainable params: 0
'''

model.save('./_save/keras26_1_save_model.h5')

# model.compile(loss='mse',optimizer='adam')
# model.fit(x_train,y_train,epochs=10)

# loss=model.evaluate(x_test,y_test)
# print('loss :',loss)




