# 이제 분류할거임
# 이진분류도 다중분류에 속함
# 이전과 다른 부분은 y가 0과 1이라는 것
'''
이진분류때
model.add(Dense(1, activation='sigmoid')) # 마지막 레이어에 sigmoid 주기 #마지막을 고쳐주면됨
model.compile(loss='binary_crossentropy'
'''
#과제=> 파이썬 책에 리스트 딕셔너리 튜플에 대해 공부하고 메일 보내기

import numpy as np
from sklearn.datasets import load_breast_cancer #유방암 데이터 암 걸렸나 안걸렸나
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

#데이터
datasets=load_breast_cancer()
#print(datasets)
print(datasets.DESCR) #DESCR 묘사 #pandas : descibe()
print(datasets.feature_names) # 컬럼이름 , #pandas :.columns()

x=datasets['data']
y=datasets.target

print(x.shape,y.shape) #(569, 30) (569,)
#print(y) # 암t/f

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=333,train_size=0.9
)
# scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

model=Sequential()
model.add(Dense(10,activation='relu',input_dim=30))
model.add(Dense(9, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(7, activation='linear'))
model.add(Dense(1, activation='sigmoid')) # 마지막 레이어에 sigmoid 주기 #마지막을 고쳐주면됨 0-1 사이의 수로 나오니까! 밑에서 반올림해줄거임

model.compile(loss='binary_crossentropy', optimizer='adam', #'mse' 실수로 나옴 그러니까 이진분류때는 쓰지말고 binary_crossentropy를 써줌
              metrics=['accuracy','acc','mse'] #,mean_squared_error #metrics=['accuracy']를 쓰면 알아서 np.round까지 해줌/'mse'도 보고싶으면 메트릭스에서 불러와보면됨,mae도 마찬가지.. 매트릭스놈들은 가능 대신 
              ) # metrics=['accuracy'와 acc=accuracy_score(y_test,y_pre)의 acc와 같은 놈

es = EarlyStopping(monitor='val_accuracy',mode='auto', patience=20,restore_best_weights=True)

model.fit(x_train,y_train, epochs=100, batch_size=8,validation_split=0.1,verbose=1,callbacks=[es])

results=model.evaluate(x_test,y_test)
print('results :', results)
'''
loss,acc,mse 세가지 나옴
results : [0.15466229617595673, 0.9385964870452881, 0.04537041485309601]
'''


# 이분법적이니까 정확도 사용하기 (이진분류)
#회귀 아님 분류
#분류는 이중 아니면 다중(예를 들어 가위,바위,보)밖에 없음
#텐서플로 2번 문제로 분류 나옴

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score #metrics 지표들 들어있음
y_pre=np.round(model.predict(x_test)) # np.round()를 써줘서 예측 값을 반올림 해주자
# print('====================')
# print(y_test[:5])
# print(np.round(y_pre[:5]))  # 반올림은 6부터
# print('====================')

acc=accuracy_score(y_test,y_pre) # accuracy_scores는 y_test,y_pre 둘이 몇프로 맞나
print('acc :',acc)

'''
MinMaxScaler

58/58 [==============================] - 0s 1ms/step - loss: 0.5941 - accuracy: 0.8283 - acc: 0.8283 - mse: 0.1335 - val_loss: 0.0961 - val_accuracy: 0.9423 - val_acc: 0.9423 - val_mse: 0.0328
2/2 [==============================] - 0s 2ms/step - loss: 0.1240 - accuracy: 0.9825 - acc: 0.9825 - mse: 0.0226
results : [0.12400084733963013, 0.9824561476707458, 0.9824561476707458, 0.022583818063139915]
acc : 0.9824561403508771

StandardScaler

Epoch 58/100
58/58 [==============================] - 0s 1ms/step - loss: 0.2214 - accuracy: 0.9087 - acc: 0.9087 - mse: 0.0649 - val_loss: 0.0986 - val_accuracy: 0.9423 - val_acc: 0.9423 - val_mse: 0.0260
2/2 [==============================] - 0s 1ms/step - loss: 0.1431 - accuracy: 0.9474 - acc: 0.9474 - mse: 0.0411
results : [0.143097922205925, 0.9473684430122375, 0.9473684430122375, 0.04105374589562416]
acc : 0.9473684210526315

RobustScaler

Epoch 33/100
58/58 [==============================] - 0s 1ms/step - loss: 0.2116 - accuracy: 0.9239 - acc: 0.9239 - mse: 0.0631 - val_loss: 0.0705 - val_accuracy: 1.0000 - val_acc: 1.0000 - val_mse: 0.0141
2/2 [==============================] - 0s 1ms/step - loss: 0.1135 - accuracy: 0.9825 - acc: 0.9825 - mse: 0.0210
results : [0.11348025500774384, 0.9824561476707458, 0.9824561476707458, 0.021024536341428757]
acc : 0.9824561403508771

MaxAbsScaler

Epoch 37/100
58/58 [==============================] - 0s 1ms/step - loss: 0.2190 - accuracy: 0.9152 - acc: 0.9152 - mse: 0.0632 - val_loss: 0.0749 - val_accuracy: 1.0000 - val_acc: 1.0000 - val_mse: 0.0160
2/2 [==============================] - 0s 1ms/step - loss: 0.1110 - accuracy: 0.9825 - acc: 0.9825 - mse: 0.0250
results : [0.11102388054132462, 0.9824561476707458, 0.9824561476707458, 0.025020861998200417]
acc : 0.9824561403508771

'''