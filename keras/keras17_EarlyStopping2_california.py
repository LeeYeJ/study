import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.callbacks import EarlyStopping

datasets= fetch_california_housing()
x=datasets.data
y=datasets['target']
#(20640, 8) (20640,)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=678,test_size=0.3
)

model=Sequential()
model.add(Dense(7,input_dim=8))
model.add(Dense(8))
model.add(Dense(5,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(9))
model.add(Dense(1))

es= EarlyStopping(monitor='val_loss',patience=20,mode='min',restore_best_weights=True)

model.compile(loss='mae',optimizer='adam')
hist=model.fit(x_train,y_train, epochs=200,batch_size=50,validation_split=0.3,callbacks=[es])

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_pre=model.predict(x_test)
r2=r2_score(y_pre,y_test)
print('r2score :', r2)

plt.plot(hist.history['val_loss'])
plt.show()

'''
Epoch 122/200
203/203 [==============================] - 0s 1ms/step - loss: 0.5596 - val_loss: 0.5530
Epoch 123/200
203/203 [==============================] - 0s 1ms/step - loss: 0.5457 - val_loss: 0.5882
194/194 [==============================] - 0s 606us/step - loss: 0.5067
loss : 0.5067328214645386
r2score : 0.3612657252335535
'''