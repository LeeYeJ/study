# # 모델 로드
# model = load_model('./_save/keras26_3_save_model.h5')
# # 앞의 3번 파일에서 컴파일 훈련 뒤 모델을 저장하니 가중치 저장돼서 출력하면 로스값이 고정됨
'''
# 가중치는 컴파일 훈련뒤에 저장해주고
model.load_weights('./_save/keras26_5_save_weights2.h5')
#웨이트는 컴파일 훈련 꼭 해줘야함 왜? 왜냐면 웨이트 값만 뽑아오니까 로드할땐 훈련할때 사용했던 지표로 컴파일을 해줘야됨!
'''
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model,load_model
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
# print(np.min(x),np.max(x)) #0.0 711.0 (0~711까지)
scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler.fit(x_train) # fit의 범위가 x_train이다
# x_train=scaler.transform(x_train) #변환시키라
x_train = scaler.fit_transform(x_train) #위에 두줄을 한줄로 쓸수있다
x_test=scaler.transform(x_test) # x_train의 범위에 맞춰서 변환해준다. 그래서 fit은 할 필요 없음
# print(np.min(x_test),np.max(x_test)) 

# 함수형 모델
input1 = Input(shape=(13,)) # 스칼라 13개 벡터 1개
dense1 = Dense(30)(input1) #(input1) 이 레이어가 어디서 왔는지
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1,outputs=output1)

# 가중치 로드
# model.load_weights('./_save/keras26_5_save_weights1.h5')
# 모델 다음 저장된 1번은 초기 랜덤값은 저장되어있음
######################################################

# 가중치는 컴파일 훈련뒤에 저장해주고
model.load_weights('./_save/keras26_5_save_weights2.h5')
#웨이트는 컴파일 훈련 꼭 해줘야함 왜? 왜냐면 웨이트 값만 뽑아오니까 로드할땐 훈련할때 사용했던 지표로 컴파일을 해줘야됨!

#컴파일 훈련
model.compile(loss='mae',optimizer='adam')
# model.fit(x_train,y_train,epochs=10)

#평가
loss=model.evaluate(x_test,y_test)
print('loss :',loss)




