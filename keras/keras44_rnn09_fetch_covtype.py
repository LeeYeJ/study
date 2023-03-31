#분류
#캐글, 따릉이, 디아벳 최대 업로드수 해서 등수까지 스냅샷 찍어서 금요일 주말 내내 제출
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

datasets=fetch_covtype() # 인터넷에서 가져와서 내 로컬에 저장되는거임. 만약 엉키면(에러) 파일 경로 찾아서 직접 삭제해줘야됨 / 사이킷럭 삭제시 cmd 창에 uninstall
x=datasets.data
y=datasets['target']

print(x.shape,y.shape) # (581012, 54) (581012,)
print(np.unique(y))  # [1 2 3 4 5 6 7] # 1부터 시작하는데 원핫인코딩(판다스,사이킷런,케라스)할때의 차이를 보고 사용 
print(datasets.DESCR)

# 원핫인코딩
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y= y.reshape(-1,1)
y=encoder.fit_transform(y).toarray()
#데이터 분리
x_train,x_test,y_train,y_test=train_test_split(
    x,y, shuffle=True, random_state=2000,train_size=0.9
)

print(x_train.shape) #
print(x_test.shape)
# scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

x_train= x_train.reshape(522910,9,6)
x_test= x_test.reshape(58102,9,6)

model = Sequential()
model.add(LSTM(16,input_shape=(9,6),activation='linear'))
model.add(Dense(9,activation='relu'))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_acc',mode='auto',patience=50)
model.fit(x_train,y_train,epochs=30,batch_size=5000,validation_split=0.1,callbacks=[es])

results=model.evaluate(x_test,y_test)
print('results :', results)

y_pre= model.predict(x_test)

y_test_acc=np.argmax(y_test, axis=1) 
# print(y_test_acc) 
y_pre=np.argmax(y_pre,axis=1)
# print(y_pre) 

acc=accuracy_score(y_pre,y_test_acc)   
print('acc :', acc)

'''
CNN모델
results : [0.6414103507995605, 0.7254827618598938]
acc : 0.7254827716773949

rnn모델
results : [0.7014074325561523, 0.7070668935775757]
acc : 0.7070668823792641
'''