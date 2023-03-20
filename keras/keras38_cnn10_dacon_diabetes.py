

#https://dacon.io/competitions/open/236068/mysubmission?isSample=1
#당뇨대회

from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Conv2D,Flatten
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
# print(x.columns)
# print(y)
# print(x.shape) # (652, 8)
# print(y.shape) # (652,)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=4545451,train_size=0.9,
)

print(x_train.shape) #(586, 8)
print(x_test.shape) #(66, 8)
print(test_csv.shape) #(116, 8)

# scaler= MinMaxScaler(
    # ) # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)


#test 파일도 스케일링 해줘야됨!!!!!!!!!
test_csv=scaler.transform(test_csv)

x_train= x_train.reshape(586,8,1,1)
x_test= x_test.reshape(66,8,1,1)
test_csv = test_csv.reshape(116,8,1,1) # test파일도 모델에서 돌려주니까 리쉐잎 해줘야됨.

model = Sequential()
model.add(Conv2D(7,(2,1),input_shape=(8,1,1)))
model.add(Conv2D(8,(2,1),activation='relu'))
model.add(Flatten())
model.add(Dense(9,activation='relu'))
model.add(Dense(6))
model.add(Dense(1))

# input1 = Input(shape=(8,))
# modell = Dense(10, activation='relu')(input1)
# model2 = Dense(9, activation='linear')(modell)
# model3 = Dense(9, activation='relu')(model2)
# model4 = Dense(7, activation='linear')(model3)
# output1 = Dense(1, activation='sigmoid')(model4)
# model= Model(inputs=input1,outputs=output1)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc']) #loss값에 대해 각 metrics=['acc','mse'] 를 확인할수있다.

es=EarlyStopping(monitor='val_loss',mode='auto',patience=190,restore_best_weights=True)

model.fit(x_train,y_train,epochs=500,batch_size=5,validation_split=0.1,callbacks=[es])

results=model.evaluate(x_test,y_test)
print('results:', results)

y_pre=np.round(model.predict(x_test))
acc=accuracy_score(y_pre,y_test)
print('acc:',acc)

y_sub=np.round(model.predict(test_csv))
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
submission['Outcome']=y_sub

submission.to_csv(path_save+'submisssion_03131820.csv')

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
'''
MinMaxScaler 

Epoch 204/500
112/112 [==============================] - 0s 982us/step - loss: 0.3135 - acc: 0.8600 - val_loss: 0.5770 - val_acc: 0.7742
2/2 [==============================] - 0s 1ms/step - loss: 0.5601 - acc: 0.7879
results: [0.5600827932357788, 0.7878788113594055]
acc: 0.7878787878787878

StandardScaler 

Epoch 204/500
112/112 [==============================] - 0s 982us/step - loss: 0.3135 - acc: 0.8600 - val_loss: 0.5770 - val_acc: 0.7742
2/2 [==============================] - 0s 1ms/step - loss: 0.5601 - acc: 0.7879
results: [0.5600827932357788, 0.7878788113594055]
acc: 0.7878787878787878

RobustScaler 

Epoch 215/500
112/112 [==============================] - 0s 1ms/step - loss: 0.2519 - acc: 0.8887 - val_loss: 0.7757 - val_acc: 0.7258
2/2 [==============================] - 0s 1ms/step - loss: 0.6866 - acc: 0.6970
results: [0.6866012811660767, 0.6969696879386902]
acc: 0.696969696969697

MaxAbsScaler

Epoch 252/500
112/112 [==============================] - 0s 987us/step - loss: 0.3972 - acc: 0.8312 - val_loss: 0.5502 - val_acc: 0.7419
Epoch 253/500
112/112 [==============================] - 0s 1ms/step - loss: 0.3991 - acc: 0.8007 - val_loss: 0.6004 - val_acc: 0.7903
2/2 [==============================] - 0s 950us/step - loss: 0.5528 - acc: 0.6970
results: [0.5528364777565002, 0.6969696879386902]
acc: 0.696969696969697

CNN
Epoch 218/500
106/106 [==============================] - 0s 1ms/step - loss: 5.5026 - acc: 0.6433 - val_loss: 4.9674 - val_acc: 0.6780
3/3 [==============================] - 0s 1ms/step - loss: 0.4606 - acc: 0.7273
results: [0.46064555644989014, 0.7272727489471436]
acc: 0.7272727272727273


'''





