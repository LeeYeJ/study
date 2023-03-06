#+써밋

# 데이터 따릉이 문제풀이
# id는 데이터 아님 -> 색인.
# 테스트 데이터는 프레딕트에서만 쓰고 트레인 데이터는 분리해서 훈련해준다.
# 중단점은 실행시키고 싶은 지점의 다음 실행코드 앞에 중단점 찍어준다. 그럼 그 전까지 실행
# 테스트 csv의 결측값 건들면 안됨?

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # mean_squared_error는 mse , rmse는 변수로 따로 만들어주면 될듯 저 함수 이용해서
import pandas as pd # 자료형, 전처리 / 판다스 자료형은 인덱스 헤더 가지고있다

#1.데이터
path = './_data/ddarung/' # 폴더 위치까지 경로 지정 .은 현재 폴더 /는 하위, 여기서 .은 study

# path 경로의 train_csv 데이터 땡겨와 변수 지정 (혹은 train_csv= pd.read_csv('./_data/ddarung/train.csv')로 표현), index_col 인덱스 컬럼은 뭔지 나타내는것
train_csv= pd.read_csv(path + 'train.csv', index_col=0)  #index_col: 각 행(row)의 이름이 위치한 열(column)을 지정. 기본값은 None
print(train_csv)
print(train_csv.shape) #(1459, 10) # 즉 첫줄 이름은 컬럼명(헤더)으로 판단, 데이터로 치지 않으니 11이 아니고 10열

test_csv= pd.read_csv(path + 'test.csv', index_col=0)  #count는 y값이다. 나머지 데이터가 x값
print(test_csv)
print(test_csv.shape) #(715, 9)

#=================================================================

print(train_csv.columns) # 컬럼명들 나옴
'''
Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count '],
      dtype='object')
'''

print(train_csv.info())
'''
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    1459 non-null   int64
 1   hour_bef_temperature    1457 non-null   float64
 2   hour_bef_precipitation  1457 non-null   float64 # 2개가 결측치다
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64 # 최소 117개 이상의데이터가 지워진다. 
 9   count                   1459 non-null   float64 #위의 이유는 행에 컬럼 하나가 결측치라도 행 하나가 다 지워진다.
'''

print(train_csv.describe())
'''
      hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5       count 
count  1459.000000           1457.000000             1457.000000         1450.000000        1457.000000          1457.000000     1383.000000    1369.000000     1342.000000  1459.000000
mean     11.493489             16.717433                0.031572            2.479034          52.231297          1405.216884        0.039149      57.168736       30.327124   108.563400
std       6.922790              5.239150                0.174917            1.378265          20.370387           583.131708        0.019509      31.771019       14.713252    82.631733
min       0.000000              3.100000                0.000000            0.000000           7.000000            78.000000        0.003000       9.000000        8.000000     1.000000
25%       5.500000             12.800000                0.000000            1.400000          36.000000           879.000000        0.025500      36.000000       20.000000    37.000000
50%      11.000000             16.600000                0.000000            2.300000          51.000000          1577.000000        0.039000      51.000000       26.000000    96.000000
75%      17.500000             20.100000                0.000000            3.400000          69.000000          1994.000000        0.052000      69.000000       37.000000   150.000000
max      23.000000             30.000000                1.000000            8.000000          99.000000          2000.000000        0.125000     269.000000       90.000000   431.000000
mean -> 평균
std -> ?
'''
print(type(train_csv)) #데이터 타입 <class 'pandas.core.frame.DataFrame'>
###########################결측치 처리################################
#특정값을 넣어도 되지만 통으로 빼도 됨

# 결측치 처리 1. 제거
print(train_csv.isnull().sum()) # 데이터를 t/f로 바꾼다. t가 결측치 /true의 합계/ 즉 결측치 확인 가능
train_csv=train_csv.dropna() # na nan 나오면 결측치/ 결측치 제거
print(train_csv.isnull().sum())
'''
hour                      0
hour_bef_temperature      0
hour_bef_precipitation    0
hour_bef_windspeed        0
hour_bef_humidity         0
hour_bef_visibility       0
hour_bef_ozone            0
hour_bef_pm10             0
hour_bef_pm2.5            0
count                     0  이렇게 다 지워짐
'''

################# train_csv에서 x와 y 데이터 분리####################
x= train_csv.drop(['count'], axis=1) # count 열을 제거해준다. ,(['count']) 두개 이상은 리스트다., axis는 열?
print(x)


y = train_csv['count']
print(y)
#############################################################

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=3507,train_size=0.7
)
#전체 train 사이즈에서 0.7만큼인 1021개의 데이터가 train 나머지가 test <=이 데이터는 모두 train_csv 데이터이다.
print(x_train.shape, x_test.shape) # (1021, 9) (438, 9) -> (929, 9) (399, 9) 결측치 삭제 값

print(y_train.shape,y_test.shape) # (1021,) (438,) -> (929,) (399,) 결측치 삭제 값

#2. 모델구성

model = Sequential()
model(Dense(9,input_dim=9))
model(Dense(1))


#3.컴파일 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=500,batch_size=30, verbose=1)

#4.평가 예측

loss=model.evaluate(x_test,y_test)
print('loss : ',loss)

y_predict= model.predict(x_test.sum(axis=1))

print(x_test.shape,y_test.shape, y_predict.shape)

r2 = r2_score(y_test,y_predict)
print('r2 스코어:',r2)

def RMSE(y_test,y_predict): #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test,y_predict)) # mse에 np.sqrt() 루트 씌어줌
rmse = RMSE(y_test,y_predict) # 함수 사용

print('RMSE :', rmse)

########submit.csv를 만들자############
#print(test_csv.isnull().sum()) 여기도 결측치가 있다
y_submit=model.predict(test_csv)
print(y_submit)

submisson = pd.read_csv(path + 'submission.csv',index_col=0)
print(submisson) #nan자리에 값을 넣어주자
submisson['count'] = y_submit #count 컬럼에 count 값을 넣어주자
print(y_submit)

#파일로 바꿔서 저장해주자
submisson.to_csv(path + 'submit_0306_0447.csv') #저장할때는 to_csv 읽어올땐 read_csv
# 만들어진 파일을 첨부해서 제출한다. 메모는 0306_0449 재출