####### 튜닝 ###########
# 데이터 크기의 조절
# 트레인 데이터 조절
# 배치조절과 에포의 조절

'''
각각 자정 넘기 전까지 제출

메일 제목: 이예지 [삼성 1차] 60,350,07원 (둘째자리까지만)
메일 제목: 이예지 [삼성 2차] 60,350,07원

첨부파일: keras53_samsung1_LYJ_submit.py 
첨부파일: keras53_samsung1_LYJ_submit.py

가중치 _save/samsung/ keras53_samsung2_LYJ.h5 / hdf5
가중치 _save/samsung/ keras53_samsung4_LYJ.h5 / hdf5

'''

# 타임스텝스가 정해지면 shuffle? randomstate? 해도 괜찮다? 오히려 순차적으로 하는것보다 과적합이 방지된다.

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Input, Dense, LSTM,Dropout, Reshape, Conv1D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score,mean_squared_error

path = './_data/시험/'
path_save = './_save/samsung/'

datasets1 = pd.read_csv(path+'삼성전자 주가3.csv',encoding='cp949',index_col=0) 

print(datasets1.shape) #(3260, 16)

datasets2 = pd.read_csv(path + '현대자동차2.csv',encoding='cp949',index_col=0)
print(datasets2.shape) # (3140, 16)

datasets1 = datasets1[::-1] # 역순으로 모든 요소 출력 (start와 end는 생략)
datasets2 = datasets2[::-1]

print(datasets1.head() )
print(datasets2.head())

print(type(datasets1), type(datasets2)) #<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.frame.DataFrame'>

print(datasets1.isnull().sum()) # 금액과 거래량에 결측치 3씩 있음
print(datasets2.isnull().sum()) # 결측치 없음

x1 = np.array(datasets1.drop(['신용비','외인(수량)','프로그램','외인비','전일비','금액(백만)','개인','기관','외국계'],axis=1))

x2 = np.array(datasets2.drop(['시가','외인(수량)','프로그램','외인비','전일비','금액(백만)','개인','기관','외국계'],axis=1))

y = np.array(datasets2['시가'])

x1 = np.char.replace(x1.astype(str), ',', '').astype(np.float64)
x2 = np.char.replace(x2.astype(str), ',', '').astype(np.float64)
y = np.char.replace(y.astype(str), ',', '').astype(np.float64)

# y_submit = y_submit.fillna(y_submit.median())
# y_submit = y_submit.fillna(y_submit.median())
x1 = x1[833:]
x2 = x2[893:]
y = y[893:]


print(x1.shape,x2.shape,y.shape) #(1207, 7) (1207, 7) (1207,)

x1 = np.char.replace(x1.astype(str), ',', '').astype(np.float64)
x2 = np.char.replace(x2.astype(str), ',', '').astype(np.float64)
y = np.char.replace(y.astype(str), ',', '').astype(np.float64)

# x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(
#     x1,x2,y, shuffle= False, train_size=0.7 
# )

_,x1_test,_,x2_test,_,y_test=train_test_split(
    x1,x2,y, shuffle= False, train_size=0.7 
)
x1_train,x2_train,y_train=x1,x2,y
print(x1_train.shape,x1_test.shape,x2_train.shape,x2_test.shape,y_train.shape,y_test.shape) #(1207, 7) (363, 7) (1207, 7) (363, 7) (1207,) (363,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # 여기서 어레이 형태로 해서 아래 리쉐잎때 변환안해줘도됨
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler.fit_transform(x2_train)
x2_test = scaler.transform(x2_test)

print(x1_train.shape,x1_test.shape,x2_train.shape,x2_test.shape,y_train.shape,y_test.shape)


timesteps=3

pred1 = x1_test[-3:].reshape(1,3,7)
pred2 = x2_test[-3:].reshape(1,3,7)
print(pred1.shape, pred2.shape) #(1, 5, 7) (1, 5, 7)

print(pred2)

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps -1):
        subset = dataset[i:(i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

x1_train = split_x(x1_train,timesteps)
x1_test = split_x(x1_test,timesteps)

x2_train = split_x(x2_train,timesteps)
x2_test = split_x(x2_test,timesteps)

print(x1_train.shape,x1_test.shape) # (838, 5, 7) (357, 5, 7)
print(x2_train.shape,x2_test.shape) # (838, 5, 7) (357, 5, 7)

y_train = y_train[4:]
y_test = y_test[4:]
print(y_train.shape,y_test.shape) # (838,) (357,)

from sklearn.metrics import accuracy_score,mean_squared_error,r2_score

model = load_model('./_save/samsung/keras53_samsung4_LYJ.h5')

y_predict = model.predict([pred1,pred2])

loss = model.evaluate([x1_test,x2_test],y_test)
print('loss :', loss)

# print("%.2f" % y_predict)
print(np.round(y_predict,2))
   





