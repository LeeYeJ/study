# 3,9는 데이터가 너무 적으니까 제거해주는 방법을 써도됨 
# 이유는 다른 값들이 3,9로 튀지 않기 위해...

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler, LabelEncoder
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

# 라벨인코더 (와인 분류 문자를 수치로 0,1)
le= LabelEncoder()
le.fit(train_csv['type'])
train_csv['type'] = le.transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])
print(test_csv) # 0과 1로 바뀜

# 결측치 제거
print(train_csv.isnull().sum()) # 없음

x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
# test_csv = test_csv.drop(['type'], axis=1)

print(x.shape, y.shape) #(5497, 12) (5497,)
print(np.unique(y)) # [3 4 5 6 7 8 9]

# y = to_categorical(y) 얘는 0부터 카운트해서 위 3부터 시작하는 라벨인 경우
# print(y)

y = pd.get_dummies(y)
print(type(y))
y = np.array(y)
print(type(y))


print(y)

# y=np.delete(y, 0, axis=1)
# y=np.delete(y, 0, axis=1)
# y=np.delete(y, 0, axis=1) 
# print('============')
# print(y)


x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True, random_state=8778778, train_size=0.9, 
    #stratify=y
)


#스케일러
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
test_csv=scaler.transform(test_csv)

input1 = Input(shape=(12,))
dense1 = Dense(50,activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(90)(drop1)
# drop2 = Dropout(0.2)(dense2)
dense3 = Dense(80)(dense2)
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


es = EarlyStopping(monitor='acc',mode='auto',patience=1000, restore_best_weights=True)

mcp = ModelCheckpoint(monitor='acc', mode='auto',save_best_only=True,
                      filepath="".join([filepath, 'k28_', date,'_',filename ]))

model.fit(x_train,y_train, epochs=100000, batch_size=100,validation_split=0.2,verbose=1,callbacks=[es,mcp])

results=model.evaluate(x_test,y_test)
print('results :', results)

y_pre= model.predict(x_test)
print(y_pre.shape) #(275, 7)

y_pre=np.argmax(y_pre,axis=1)
print(y_pre.shape)
print(y_pre)

y_test_acc=np.argmax(y_test, axis=1) 
print(y_test_acc.shape)

acc=accuracy_score(y_pre,y_test_acc)
print('acc:',acc)

y_sub=np.argmax(model.predict(test_csv), axis=1)
y_sub += 3


submission['quality']=y_sub


print(y_sub)

submission.to_csv(path_save+'submisssion_03142015.csv')













