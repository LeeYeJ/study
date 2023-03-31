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

print(datasets1.shape) #(2040, 16)

datasets2 = pd.read_csv(path + '현대자동차2.csv',encoding='cp949',index_col=0)
print(datasets2.shape) # (2100, 16)

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

x1 = x1[840:]
x2 = x2[900:]
y = y[840:]


print(x1.shape,x2.shape,y.shape) #(1200, 11) (1200, 11) (1200,)

x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(
    x1,x2,y, shuffle= False, train_size=0.8
)

pred1 = x1_test[-3:].reshape(1,3,11)
pred2 = x2_test[-3:].reshape(1,3,11)
print(pred1.shape, pred2.shape) #(1, 3, 11) (1, 3, 11)
                                                                                                    
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

print(x1_train.shape,x1_test.shape) # (1077, 3, 11) (117, 3, 11)
print(x2_train.shape,x2_test.shape) # (1077, 3, 11) (117, 3, 11)

y_train = y_train[timesteps:]
y_test = y_test[timesteps:]
print(y_train.shape,y_test.shape) # (1077,) (117,)

from sklearn.metrics import accuracy_score,mean_squared_error,r2_score

model = load_model('./_save/samsung/keras53_samsung2_LYJ.h5')

y_predict = model.predict([pred1,pred2])

loss = model.evaluate([x1_test,x2_test],y_test)
print('loss :', loss)

print("%.2f" % y_predict)