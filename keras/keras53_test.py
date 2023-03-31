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

x1 = np.array(datasets1.drop(['신용비','외인(수량)','프로그램','외인비','전일비'],axis=1))

x2 = np.array(datasets2.drop(['시가','외인(수량)','프로그램','외인비','전일비'],axis=1))

y = np.array(datasets2['시가'])

x1 = np.char.replace(x1.astype(str), ',', '').astype(np.float64)
x2 = np.char.replace(x2.astype(str), ',', '').astype(np.float64)
y = np.char.replace(y.astype(str), ',', '').astype(np.float64)

# y_submit = y_submit.fillna(y_submit.median())
# y_submit = y_submit.fillna(y_submit.median())
x1 = x1[833:]
x2 = x2[893:]
y = y[893:]


print(x1.shape,x2.shape,y.shape) #(1207, 11) (1207, 11) (1207,)

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
print(x1_train.shape,x1_test.shape,x2_train.shape,x2_test.shape,y_train.shape,y_test.shape) #(844, 11) (363, 11) (844, 11) (363, 11) (844,) (363,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # 여기서 어레이 형태로 해서 아래 리쉐잎때 변환안해줘도됨
x1_train = scaler.fit_transform(x1_train).reshape(-1,11,1)
x1_test = scaler.transform(x1_test).reshape(-1,11,1)
x2_train = scaler.fit_transform(x2_train).reshape(-1,11,1)
x2_test = scaler.transform(x2_test).reshape(-1,11,1)

print(x1_train.shape,x1_test.shape,x2_train.shape,x2_test.shape,y_train.shape,y_test.shape)


timesteps=5

pred1 = x1_test[-5:].reshape(1,5,11)
pred2 = x2_test[-5:].reshape(1,5,11)
print(pred1.shape, pred2.shape) #(1, 5, 11) (1, 5, 11)

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

print(x1_train.shape,x1_test.shape) # (838, 5, 11) (357, 5, 11)
print(x2_train.shape,x2_test.shape) # (838, 5, 11) (357, 5, 11)

y_train = y_train[6:]
y_test = y_test[6:]
print(y_train.shape,y_test.shape) # (838,) (357,)
                       
                                          
#2-1 모델1
input1 = Input(shape=(5,11))
dense1 = LSTM(30,return_sequences=True, name='stock1')(input1)
lstm1 = LSTM(16)(dense1)
dense13 = Dense(30,name='weather3234',activation='selu')(lstm1)
dense14 = Dense(15,name='weather4234')(dense13)
dense3 = Dense(30,activation='relu', name='stock3')(dense14)
output1 = Dense(10, name='output1')(dense3) # 아웃풋 노드의 갯수는 자유롭게 줘도됨 오히려 적으면 많이 축소됨 모델 1,2는 합병된 전체 모델의 히든레이어임

#2-2 모델2
input2 = Input(shape=(5,11))
dense1 = LSTM(30,return_sequences=True, name='stock01')(input1)
lstm1 = LSTM(16)(dense1)
dense13 = Dense(30,name='weather3',activation='relu')(lstm1)
dense14 = Dense(15,name='weather4')(dense13)
dense135 = Dense(30,name='weather399',activation='relu')(dense14)
dense146 = Dense(15,name='weather499')(dense135)
output2 = Dense(10,name='output2')(dense146)


#2-4 머지 # 히든이니까 값 크게줘도됨
from tensorflow.keras.layers import concatenate 
merge1 = concatenate([output1, output2],name='mg1') # 두 모델의 아웃풋을 합병한다. / 두개 이상이니까 리스트 형태로 받는다.
merge2 = Dense(80, activation='relu', name='mg2')(merge1)
# resh1 = Reshape(target_shape=(8,10))(merge2)
# conv1 = Conv1D(10,8)(resh1)
# resh2 = Flatten()(conv1)
dense138 = Dense(60,name='weather3124',activation='relu')(merge2)
dense149 = Dense(70,name='weather45235')(dense138)
merge3 = Dense(120, activation='relu', name='mg3')(dense149)#(resh2)
last_output = Dense(1,name='last_output')(merge3)

model = Model(inputs=[input1,input2],outputs = last_output)
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss ='mse', optimizer='adam')
es = EarlyStopping(monitor='loss',mode='auto',patience=20,restore_best_weights=True)
model.fit([x1_train,x2_train],y_train, epochs=120,batch_size=10,verbose=1)

model.save('./_save/samsung/keras53_samsung4_model07_.h5')

from sklearn.metrics import accuracy_score,mean_squared_error,r2_score

# model = load_model('./_save/samsung/keras53_save_model_5.h5')

y_predict = model.predict([pred1,pred2])

# def RMSE(y_test,y_predict): 
#     return np.sqrt(mean_squared_error(y_test,y_predict)) 
# rmse = RMSE(y_test, y_predict)                           
# print("RMSE : ", rmse)

loss = model.evaluate([x1_test,x2_test],y_test)
print('loss :', loss)

# print("%.2f" % y_predict)
print(np.round(y_predict,2))


'''
1 - 178024.95
2 - 177411.88
3 - 173122.19
4 - 178792.48
5 - 175895.31
6 - 176421.08
7 - 182257.52
8 - [[178260.6]]
'''



