from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler , StandardScaler #preprocessing 전처리, MinMaxScaler 정규화, tandardScaler 평균점을 중심으로 데이터를 모은다.
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

# 훈련데이터만 정규화한다.

# #2.모델 구성
# model=Sequential()
# model.add(Dense(1,input_dim=13))

# model.compile(loss='mse',optimizer='adam')
# model.fit(x_train,y_train,epochs=10)

# loss=model.evaluate(x_test,y_test)
# print('loss :',loss)




