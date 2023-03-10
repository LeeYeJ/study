import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pylab as plt

path='./_data/ddarung/'
path_save='./_save/ddarung/'

train_csv=pd.read_csv(path+'train.csv', index_col=0)

print(train_csv.shape) #(1459, 10)

test_csv=pd.read_csv(path +'test.csv', index_col=0)

print(test_csv.shape) #(715, 9)

#결측치 제거
print(train_csv.isnull().sum())
train_csv=train_csv.dropna()
print(train_csv.isnull().sum())

#데이터분리!!!!!!!!!!!!!!! 외워 좀!! 
x=train_csv.drop(['count'],axis=1)
y=train_csv['count']

#데이터분리

x_train,x_test,y_train,y_test=train_test_split(
 x,y,shuffle=True,random_state=12334,test_size=0.1
)

#모델
model = Sequential()
model.add(Dense(6, input_dim = 9))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(1))

#컴파일
es=EarlyStopping(monitor='val_loss',mode='min',patience=30,restore_best_weights=True)

model.compile(loss='mae',optimizer='adam')
hist=model.fit(x_train,y_train,epochs=200, batch_size=50,validation_split=0.2,callbacks=[es])

loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_pre=model.predict(x_test)
r2=r2_score(y_test,y_pre)
print('r2:',r2)

def RMSE(y_pre,y_test):
    return np.sqrt(mean_squared_error(y_pre,y_test))
rmse=RMSE(y_pre,y_test)

print('rmse:',rmse)

y_sub=model.predict(test_csv)
submission = pd.read_csv(path + 'submission.csv', index_col = 0)
submission['count'] = y_sub
submission.to_csv(path_save + 'submit_0308_1940.csv')

plt.plot(hist.history['val_loss'])
plt.show()

'''
Epoch 162/200
20/20 [==============================] - 0s 2ms/step - loss: 40.7933 - val_loss: 40.3410
Epoch 163/200
20/20 [==============================] - 0s 2ms/step - loss: 39.8489 - val_loss: 41.9108
5/5 [==============================] - 0s 642us/step - loss: 43.4774
loss: 43.4774284362793
r2: 0.5020065354260206

############################

Epoch 136/200
20/20 [==============================] - 0s 2ms/step - loss: 39.7032 - val_loss: 37.7733
Epoch 137/200
20/20 [==============================] - 0s 2ms/step - loss: 39.5928 - val_loss: 37.3271
5/5 [==============================] - 0s 749us/step - loss: 43.3768
loss: 43.37683868408203
r2: 0.5130379170439884
rmse: 57.66937949994228
'''



