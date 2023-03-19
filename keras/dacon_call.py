from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Dropout, pooling
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

path='./_data/dacon_call/'
path_save='./_save/dacon_call/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission=pd.read_csv(path+'sample_submission.csv', index_col=0)

print(train_csv.isnull().info()) # 결측치 없음



x = train_csv.drop(['전화해지여부'],axis=1)
y= train_csv['전화해지여부']

x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=True, random_state=54547, train_size=0.9
)

print(x_train.shape, y_train.shape) # (27180, 12) (27180,)
print(x_test.shape, y_train.shape) # (3020, 12) (27180,)

model = Sequential()
model.add(Dense(50,input_dim=12))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(9,activation='relu'))
model.add(Dense(8))
model.add(Dense(9,activation='relu'))
model.add(Dense(1,activation='softmax'))

import time
start_time=time.time()

model.compile(loss='binary_crossentropy', optimizer='adam')

import datetime 
date=datetime.datetime.now()
print(date)
date=date.strftime('%m%d_%H%M')
print(date)

es=EarlyStopping(monitor='acc', mode='auto',patience=200,restore_best_weights=True)

filepath='./_save/MCP/keras28/'
filename='{epoch:04d}-{val_loss:4f}.hdf5'

mcp = ModelCheckpoint(monitor='acc',mode='auto',save_best_only=True,
                      filepath=''.join([filepath,'k_28',date,'-',filename]))
model.fit(x_train,y_train, epochs=500, batch_size =50, validation_split=0.2)
end_time = time.time()

results= model.evaluate(x_test,y_test)
print('results :', results)

y_pred=model.predict(x_test)
y_test_acc=np.argmax(y_test,axis=1) 
# print(y_test_acc) 
y_pred=np.argmax(y_pred,axis=1)
# print(y_pre) 

acc=accuracy_score(y_pred,y_test_acc)   
print('acc :', acc)

print('time:', round(end_time-start_time,2))



























