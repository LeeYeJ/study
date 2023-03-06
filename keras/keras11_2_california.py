from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1.데이터
datasets = fetch_california_housing()
x = datasets.data
y= datasets.target

print(x.shape,y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.9, random_state=123
)
# [실습] 
# R2 0.55~0.6  이상

model = Sequential()
model.add(Dense(6,input_dim=8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=700, batch_size=50)

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

r2= r2_score(y_test, y_predict)
print('r2 score: ', r2)

'''
x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.9, random_state=123
)

model = Sequential()
model.add(Dense(6,input_dim=8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=700, batch_size=50)

Epoch 699/700
372/372 [==============================] - 0s 948us/step - loss: 0.6001
Epoch 700/700
372/372 [==============================] - 0s 936us/step - loss: 0.5970
65/65 [==============================] - 0s 788us/step - loss: 0.6095
loss : 0.6095120310783386
65/65 [==============================] - 0s 721us/step
r2 score:  0.562578411161635
'''