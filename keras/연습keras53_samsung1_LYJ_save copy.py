# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞추기

#각각 데이터에서 컬럼 7개 이상 추출 ( 그중 거래량은 반드시 들어갈것)
#timesteps와 featuer은 알아서 자르기
#제공된 데이터 외 추가 데이터 사용 금지

#1. 삼성전자 28일 종가 맞추기(점수 배점 0.3)
#2. 삼성전자 29일 시가 맞추기(점수 배점 0,7)

'''
각각 자정 넘기 전까지 제출

메일 제목: 이예지 [삼성 1차] 60,350,07원 (둘째자리까지만)
메일 제목: 이예지 [삼성 2차] 60,350,07원

첨부파일: keras53_samsung2_LYJ_submit.py 
첨부파일: keras53_samsung2_LYJ_submit.py

가중치 _save/samsung/ keras53_samsung2_LYJ.h5 / hdf5
가중치 _save/samsung/ keras53_samsung4_LYJ.h5 / hdf5

'''

# 타임스텝스가 정해지면 shuffle? randomstate? 해도 괜찮다? 오히려 순차적으로 하는것보다 과적합이 방지된다.
# 하면서 막혔던 부분은 
'''
1. x1 = np.char.replace(x1.astype(str), ',', '').astype(np.float64)

- 이 코드는 numpy 라이브러리를 이용하여 배열(array)에서 문자열을 제거하고, 그 결과를 실수형(float64)으로 변환하는 코드입니다.

  해당 코드를 구성하는 내용을 자세히 살펴보면 다음과 같습니다.

  x1.astype(str): 배열 x1의 값을 문자열(string)로 변환합니다. astype() 함수는 배열의 데이터 타입을 지정된 타입으로 변환합니다. 
  여기서는 str로 변환했으므로, x1 배열의 각 요소는 문자열이 됩니다.

  np.char.replace(x1.astype(str), ',', ''): np.char.replace() 함수를 사용하여 문자열에서 쉼표(,)를 제거합니다. 
  이 함수는 문자열에서 특정 문자나 문자열을 다른 문자나 문자열로 대체합니다. 여기서는 ,를 빈 문자열('')로 대체하여 제거하였습니다.

  astype(np.float64): 제거된 쉼표가 없어진 문자열을 다시 실수형(float64)으로 변환합니다. 
  이렇게 변환된 배열은 x1과 같은 크기와 데이터 타입을 가지며, 쉼표가 없어진 상태로 저장됩니다.

  즉, 해당 코드는 배열 x1의 요소 값 중에서 쉼표(,)를 제거하고, 그 결과를 실수형으로 변환하여 저장하는 코드입니다. 
  이 코드를 통해 데이터 전처리 과정에서 문자열에 포함된 불필요한 기호나 문자를 제거하고, 데이터를 적절한 형태로 변환할 수 있습니다.

2. 데이터 역순 추출 (train_test를 나누어 시계열로 예측하기 위해)

- datasets1 = datasets1[::-1] # 역순으로 모든 요소 출력 (start와 end는 생략)
  datasets2 = datasets2[::-1]

3. 소수점 두번쨰자리까지 출력

- print("%.2f" % y_predict[0])

  위 코드에서, %.2f는 소수점 아래 두 자리까지 출력하도록 포맷을 지정합니다

  %는 문자열 포맷팅에서 사용되는 기호 중 하나입니다. 이 기호를 사용하면 문자열 안에서 값을 동적으로 삽입할 수 있습니다.
'''

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Input, Dense, LSTM,Dropout, Reshape, Conv1D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score,mean_squared_error

path = './_data/시험 연습/'
path_save = './_save/samsung/'

datasets1 = pd.read_csv(path+'삼성전자 주가2.csv',encoding='cp949',index_col=0) 

print(datasets1.shape) #(3260, 16)

datasets2 = pd.read_csv(path + '현대자동차.csv',encoding='cp949',index_col=0)
print(datasets2.shape) # (3140, 16)

datasets1 = datasets1[::-1] # 역순으로 모든 요소 출력 (start와 end는 생략)
datasets2 = datasets2[::-1]

print(datasets1.head() )
print(datasets2.head())

print(type(datasets1), type(datasets2)) #<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.frame.DataFrame'>

print(datasets1.isnull().sum()) # 금액과 거래량에 결측치 3씩 있음
print(datasets2.isnull().sum()) # 결측치 없음

x1 = np.array(datasets1.drop(['종가','외인(수량)','프로그램','외인비','전일비'],axis=1))

x2 = np.array(datasets2.drop(['신용비','외인(수량)','프로그램','외인비','전일비'],axis=1))

y = np.array(datasets1['종가'])

x1 = np.char.replace(x1.astype(str), ',', '').astype(np.float64)
x2 = np.char.replace(x2.astype(str), ',', '').astype(np.float64)
y = np.char.replace(y.astype(str), ',', '').astype(np.float64)

# y_submit = y_submit.fillna(y_submit.median())
# y_submit = y_submit.fillna(y_submit.median())
x1 = x1[2174:]
x2 = x2[2054:]
y = y[2174:]


print


print(x1.shape,x2.shape,y.shape) #(1207, 11) (1207, 11) (1207,)

x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(
    x1,x2,y, shuffle= False, train_size=0.8 
)

pred1 = x1_test[-3:].reshape(1,3,11)
pred2 = x2_test[-3:].reshape(1,3,11)
print(pred1.shape, pred2.shape)
                                                                                                    
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler() # 여기서 어레이 형태로 해서 아래 리쉐잎때 변환안해줘도됨
# x1_train = scaler.fit_transform(x1_train)
# x1_test = scaler.transform(x1_test)
# x2_train = scaler.fit_transform(x2_train)
# x2_test = scaler.transform(x2_test)
# y_train = scaler.fit_transform(y_train)
# y_test = scaler.transform(y_test)

timesteps=3

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps ):
        subset = dataset[i:(i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x1_train = split_x(x1_train,timesteps)
x1_test = split_x(x1_test,timesteps)

x2_train = split_x(x2_train,timesteps)
x2_test = split_x(x2_test,timesteps)

# print('=============')
# print(x1_test[-1:])

print(x1_train.shape,x1_test.shape) # (962, 3, 11) (239, 3, 11)
print(x2_train.shape,x2_test.shape) # (962, 3, 11) (239, 3, 11)

y_train = y_train[timesteps:]
y_test = y_test[timesteps:]
print(y_train.shape,y_test.shape) # (962,) (239,)
                       
                                          
#2-1 모델1
input1 = Input(shape=(3,11))
dense1 = LSTM(30,activation='relu', name='stock1')(input1)
dense2 = Dense(20,activation='relu', name='stock2')(dense1)
resh1 = Reshape(target_shape=(4,5))(dense2)
lstm1 = LSTM(16,activation='relu')(resh1)
Drop1 = Dropout(0.3)(lstm1)
dense3 = Dense(30,activation='relu', name='stock3')(Drop1)
output1 = Dense(10, name='output1')(dense3) # 아웃풋 노드의 갯수는 자유롭게 줘도됨 오히려 적으면 많이 축소됨 모델 1,2는 합병된 전체 모델의 히든레이어임

#2-2 모델2
input2 = Input(shape=(3,11))
dense11 = LSTM(20,name='weather1')(input2)
dense12 = Dense(16,name='weather2')(dense11)
resh1 = Reshape(target_shape=(2,8))(dense12)
lstm1 = LSTM(16,activation='relu')(resh1)
dense13 = Dense(30,name='weather3',activation='relu')(lstm1)
dense14 = Dense(15,name='weather4')(dense13)
output2 = Dense(10,name='output2')(dense14)


#2-4 머지 # 히든이니까 값 크게줘도됨
from tensorflow.keras.layers import concatenate 
merge1 = concatenate([output1, output2],name='mg1') # 두 모델의 아웃풋을 합병한다. / 두개 이상이니까 리스트 형태로 받는다.
merge2 = Dense(80, activation='relu', name='mg2')(merge1)
# resh1 = Reshape(target_shape=(8,10))(merge2)
# conv1 = Conv1D(10,8)(resh1)
# resh2 = Flatten()(conv1)
merge3 = Dense(120, activation='relu', name='mg3')(merge2)#(resh2)
last_output = Dense(1,name='last_output')(merge3)

model = Model(inputs=[input1,input2],outputs = last_output)
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss ='mse', optimizer='adam')
es = EarlyStopping(monitor='loss',mode='auto',patience=20,restore_best_weights=True)
model.fit([x1_train,x2_train],y_train, epochs=10,batch_size=3)

model.save('./_save/samsung/keras53_save_model_3.h5')

from sklearn.metrics import accuracy_score,mean_squared_error,r2_score

# model = load_model('./_save/samsung/keras53_save_model_1.h5')

y_predict = model.predict([pred1,pred2])

# def RMSE(y_test,y_predict): 
#     return np.sqrt(mean_squared_error(y_test,y_predict)) 
# rmse = RMSE(y_test, y_predict)                           
# print("RMSE : ", rmse)

loss = model.evaluate([x1_test,x2_test],y_test)
print('loss :', loss)

print("%.2f" % y_predict)


'''
1 - 62
2 -
3 - 
'''


