# 이전 리뷰 : y값이 수치로 나오면 선형 회구 / y 값을 다중분류 할것인가

#이미지 - 가로 세로 색깔 장수 (4차원)


#트레인이고 모고 다 스케일링 해주기(테스트, 프레딕트도...)!!!!!!!

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler , StandardScaler 
#preprocessing 전처리, MinMaxScaler 정규화, tandardScaler 평균점을 중심으로 데이터를 모은다.
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
scaler.fit(x_test) 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test) # x_train의 범위에 맞춰서 변환해준다. 그래서 fit은 할 필요 없음
print(np.min(x_test),np.max(x_test)) 

#훈련데이터만 정규화한다.

#2.모델 구성
model=Sequential()
#model.add(Dense(1,input_dim=13))
model.add(Dense(1, input_shape=(13,))) # 열의 갯수만 땡겨온다 (벡터형식으로 나타내어준것)

#데이터가 3차원이면(시계열 데이터)
# (1000,100,1) -> input_shape(100,1) # 1000 데이터 갯수(행) /  100,1 (열)

# 데이터가 4차원이면 (이미지 데이터)
# (8000, 32, 32, 3) -> input_shape(32,32,3) # 8000 데이터 갯수(행) /  32,32,3 (열)

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=10)

loss=model.evaluate(x_test,y_test)
print('loss :',loss)




