#랜덤 웨이트값 던질때 음수로 주어질수도 있음. 그러면 최종값이 음수가 될수도 있음.
#모델링까지
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

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
    x,y,shuffle=True,random_state=116,train_size=0.9
)
print(x_train.shape,x_test.shape) # (9797, 8) (1089, 8)
print(y_train.shape,y_test.shape) # (9797,) (1089,)

# 모델 구성
model=Sequential()
model.add(Dense(5,input_dim=8))
model.add(Dense(6))
model.add(Dense(6,activation='relu')) #음수값 조정할때 활성화 함수나 한정화 함수로 각 층에서 던져줄때 조정해줄수있음 
model.add(Dense(5,activation='relu')) #(예를들면 음수에서 양수화해서 던져줌)
model.add(Dense(6,activation='relu'))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(1))


#컴파일 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=500,batch_size=100)

#평가
loss= model.evaluate(x_test,y_test)
print('loss :',loss)















