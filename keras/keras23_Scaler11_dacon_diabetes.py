

#https://dacon.io/competitions/open/236068/mysubmission?isSample=1
#당뇨대회

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

path='./_data/dacon_diabets/'
path_save='./_save/dacon_diabets/'

train_csv=pd.read_csv(path+'train.csv',index_col=0) #@@인덱스!

test_csv=pd.read_csv(path+'test.csv',index_col=0)

print(train_csv.isnull().sum()) #@@@@@@@@@결측치 확인!
'''
ID                          0
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
'''
#@@@@@@@@@데이터분리
x=train_csv.drop(['Outcome'],axis=1) #@@@@
y=train_csv['Outcome']
print(x.columns)
print(y)
print(x.shape) # (652, 8)
print(y.shape) # (652,)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=215312151,train_size=0.95,
)

# scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

#test 파일도 스케일링 해줘야됨!!!!!!!!!
test_csv=scaler.transform(test_csv)

model=Sequential()
model.add(Dense(10,activation='relu',input_dim=8))
model.add(Dense(9,activation='linear'))
model.add(Dense(9,activation='relu'))
model.add(Dense(7,activation='linear'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc']) #loss값에 대해 각 metrics=['acc','mse'] 를 확인할수있다.

es=EarlyStopping(monitor='val_loss',mode='auto',patience=190,restore_best_weights=True)

model.fit(x_train,y_train,epochs=500,batch_size=1,validation_split=0.1,callbacks=[es])

results=model.evaluate(x_test,y_test)
print('results:', results)

y_pre=np.round(model.predict(x_test))
acc=accuracy_score(y_pre,y_test)
print('acc:',acc)

y_sub=np.round(model.predict(test_csv))
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
submission['Outcome']=y_sub

submission.to_csv(path_save+'submisssion_03101730.csv')


'''
MinMaxScaler

Epoch 246/500
557/557 [==============================] - 0s 768us/step - loss: 0.3987 - acc: 0.8061 - val_loss: 0.5472 - val_acc: 0.7419
2/2 [==============================] - 0s 0s/step - loss: 0.4172 - acc: 0.8182
results: [0.4171835482120514, 0.8181818127632141]
acc: 0.8181818181818182

StandardScaler

Epoch 195/500
557/557 [==============================] - 0s 749us/step - loss: 0.2636 - acc: 0.8887 - val_loss: 0.9648 - val_acc: 0.7097
Epoch 196/500
557/557 [==============================] - 0s 780us/step - loss: 0.2393 - acc: 0.8941 - val_loss: 1.0744 - val_acc: 0.6774
2/2 [==============================] - 0s 1ms/step - loss: 0.4424 - acc: 0.7576
results: [0.442445307970047, 0.7575757503509521]
acc: 0.7575757575757576

RobustScaler

Epoch 197/500
557/557 [==============================] - 0s 813us/step - loss: 0.2746 - acc: 0.8725 - val_loss: 0.9573 - val_acc: 0.6129
2/2 [==============================] - 0s 0s/step - loss: 0.4254 - acc: 0.8182
results: [0.4254215955734253, 0.8181818127632141]
acc: 0.8181818181818182

MaxAbsScaler

Epoch 253/500
557/557 [==============================] - 0s 784us/step - loss: 0.3950 - acc: 0.8151 - val_loss: 0.8885 - val_acc: 0.7903
2/2 [==============================] - 0s 0s/step - loss: 0.3947 - acc: 0.8182
results: [0.39465224742889404, 0.8181818127632141]
acc: 0.8181818181818182

'''






