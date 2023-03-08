import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

print(x.shape,y.shape) #(442, 10) (442,)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=123,test_size=0.2
)
x_train,x_val,y_train,y_val=train_test_split(
    x_train,y_train,shuffle=True,random_state=66,train_size=0.6
)

#모델구성
model=Sequential()
model.add(Dense(6,input_dim=10))
model.add(Dense(5,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
hist=model.fit(x_train,y_train,epochs=100,batch_size=1,validation_data=(x_val,y_val))

plt.plot(hist.history['loss'])
plt.show()