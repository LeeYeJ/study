import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x= np.array([1,2,3])
y=np.array([4,5,6])

model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=1000, batch_size=1)



loss = model.evaluate(x,y)
print('loss :' , loss)


result = model.predict([4])
print('result : ', result)

#loss : 0.0013466635718941689
#1/1 [==============================] - 0s 71ms/step
#result :  [[3.9473035]]