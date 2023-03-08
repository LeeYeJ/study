from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#데이터
datasets=fetch_california_housing()
x=datasets.data
y=datasets['target']

print(x.shape,y.shape) #(20640, 8) (20640,)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=123,test_size=0.2
)

#모델 구성
model=Sequential()
model.add(Dense(5,input_dim=8))
model.add(Dense(6,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
hist=model.fit(x_train,y_train,epochs=500,validation_split=0.2,batch_size=100) #현 코드에서는 hist.history안에 loss, val_loss값 들어있음

plt.plot(hist.history['loss']) #에포당 로스값 출력인데 에포가 순서대로니까 굳이 명시안해도됨
plt.show() #너무 많이 돌려도 loss값이 많이 튐(과적합,오버핏)
