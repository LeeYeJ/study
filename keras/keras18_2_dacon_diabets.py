#https://dacon.io/competitions/open/236068/mysubmission?isSample=1
#당뇨대회

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

submission.to_csv(path_save+'submisssion_03101537.csv')

'''
x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=321847,train_size=0.9
)

model=Sequential()
model.add(Dense(10,activation='relu',input_dim=8))
model.add(Dense(9,activation='linear'))
model.add(Dense(9,activation='linear'))
model.add(Dense(7,activation='linear'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc','mse'])

es=EarlyStopping(monitor='val_loss',mode='auto',patience=40,restore_best_weights=True)

model.fit(x_train,y_train,epochs=500,batch_size=10,validation_split=0.1,callbacks=[es])

Epoch 499/500
53/53 [==============================] - 0s 1ms/step - loss: 0.4565 - acc: 0.7666 - mse: 0.1510 - val_loss: 0.4337 - val_acc: 0.7797 - val_mse: 0.1424
Epoch 500/500
53/53 [==============================] - 0s 1ms/step - loss: 0.4427 - acc: 0.7780 - mse: 0.1455 - val_loss: 0.4137 - val_acc: 0.7288 - val_mse: 0.1351
3/3 [==============================] - 0s 1ms/step - loss: 0.5333 - acc: 0.7576 - mse: 0.1796
results: [0.5333239436149597, 0.7575757503509521, 0.17956961691379547]
acc: 0.7575757575757576
'''

'''
x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=43879,train_size=0.9,stratify=y
)


Epoch 231/500
106/106 [==============================] - 0s 1ms/step - loss: 0.4704 - acc: 0.7647 - mse: 0.1556 - val_loss: 0.5393 - val_acc: 0.7458 - val_mse: 0.1817
3/3 [==============================] - 0s 895us/step - loss: 0.4717 - acc: 0.7727 - mse: 0.1494
results: [0.47167178988456726, 0.7727272510528564, 0.14935514330863953]
acc: 0.7727272727272727
'''

'''
x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=3445642,train_size=0.9,stratify=y
)

Epoch 190/500
176/176 [==============================] - 0s 940us/step - loss: 0.4884 - acc: 0.7533 - mse: 0.1617 - val_loss: 0.4136 - val_acc: 0.8136 - val_mse: 0.1320
3/3 [==============================] - 0s 514us/step - loss: 0.4567 - acc: 0.8182 - mse: 0.1469
results: [0.4567233622074127, 0.8181818127632141, 0.1469242423772812]
acc: 0.8181818181818182
'''

'''
x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=87984213,train_size=0.9,stratify=y
)

Epoch 114/500
176/176 [==============================] - 0s 910us/step - loss: 0.4884 - acc: 0.7514 - mse: 0.1633 - val_loss: 0.5051 - val_acc: 0.7458 - val_mse: 0.1706
3/3 [==============================] - 0s 1ms/step - loss: 0.4929 - acc: 0.7879 - mse: 0.1619
results: [0.49288085103034973, 0.7878788113594055, 0.1618848592042923]
acc: 0.7878787878787878
'''
'''
x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=3445642,train_size=0.95,stratify=y
)

Epoch 255/500
588/588 [==============================] - 1s 908us/step - loss: 0.4465 - acc: 0.7959 - mse: 0.1454 - val_loss: 0.5904 - val_acc: 0.7419 - val_mse: 0.1965
2/2 [==============================] - 0s 0s/step - loss: 0.4509 - acc: 0.7576 - mse: 0.1448
results: [0.45087605714797974, 0.7575757503509521, 0.14482203125953674]
acc: 0.7575757575757576
'''




