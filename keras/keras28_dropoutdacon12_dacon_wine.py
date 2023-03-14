from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

# 데이터 준비
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv= pd.read_csv(path +'train.csv', index_col=0)
print(train_csv.shape) #(5497, 14)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv.shape) #(1000, 13)

submission = pd.read_csv(path+'sample_submission.csv', index_col=0)

# 결측치 제거
print(train_csv.isnull().sum()) # 없음

x = train_csv.drop(['type','quality'], axis=1)
y = train_csv['quality']
test_csv = test_csv.drop(['type'], axis=1)

print(x.shape, y.shape) #(5497, 11) (5497,)
print(np.unique(y)) # [3 4 5 6 7 8 9]

y = to_categorical(y)
print(y)

y=np.delete(y, 0, axis=1)
y=np.delete(y, 0, axis=1)
y=np.delete(y, 0, axis=1) 


x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True, random_state=3444548, train_size=0.9, 
    #stratify=y
)

#스케일러
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_csv=scaler.transform(test_csv)

input1 = Input(shape=(11,))
dense1 = Dense(50,activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(90)(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(80)(drop2)
dense4 = Dense(80)(dense3)
output1 = Dense(7, activation='softmax')(dense4)
model = Model(inputs = input1, outputs= output1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

import datetime # 시간을 저장해줌
date = datetime.datetime.now() # 현재 시간
print(date) # 2023-03-14 11:15:39.585470
date = date.strftime('%m%d_%H%M') # 시간을 문자로 바꾼다 ( 월, 일, 시 ,분)
print(date) # 0314_1115

filepath='./_save/MCP/keras28/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #val_loss:4f 소수 넷째자리까지 받아와라


es = EarlyStopping(monitor='val_acc',mode='auto',patience=300, restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_acc', mode='auto',save_best_only=True,
                      filepath="".join([filepath, 'k28_', date,'_',filename ]))

model.fit(x_train,y_train, epochs=10000, batch_size=100,validation_split=0.05,verbose=1,callbacks=[es,mcp])

results=model.evaluate(x_test,y_test)
print('results :', results)

y_pre= model.predict(x_test)

y_test_acc=np.argmax(y_test, axis=1) 
 
y_pre=np.argmax(y_pre,axis=1)


acc=accuracy_score(y_pre,y_test_acc)
print('acc:',acc)

y_sub=np.argmax(model.predict(test_csv), axis=1)
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
y_sub += 3


submission['quality']=y_sub


print(y_sub)

submission.to_csv(path_save+'submisssion_03141945.csv')













