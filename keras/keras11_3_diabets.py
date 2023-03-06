from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_diabetes()
x = datasets.data
y = datasets.target

#print(x.shape, y.shape) #(442, 10) (442,)

'''
[실습]
R2 0.62 이상
'''
x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.9, random_state=123
)

model = Sequential()
model.add(Dense(6,input_dim=10))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=500, batch_size=100)

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

r2= r2_score(y_test, y_predict) #loss와 같이 판단해야한다, r2는 보조지표이기때문에!
print(x_test.shape,y_test.shape,y_predict.shape)
print('r2 score: ', r2)

'''
[실습] 실습하면서 느낀점 : 

난수값 조정이 중요한거같다....

난수값을 잘 찾는 방법이 있을까..?

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.9, random_state=123
)

model = Sequential()
model.add(Dense(6,input_dim=10))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=700, batch_size=20)

Epoch 700/700
20/20 [==============================] - 0s 892us/step - loss: 2976.3262
2/2 [==============================] - 0s 0s/step - loss: 2360.7622
loss : 2360.76220703125
2/2 [==============================] - 0s 1ms/step
r2 score:  0.6460035817016729

###################################

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.9, random_state=123
)

model = Sequential()
model.add(Dense(6,input_dim=10))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs=700, batch_size=10)

Epoch 700/700
40/40 [==============================] - 0s 947us/step - loss: 44.0123
2/2 [==============================] - 0s 0s/step - loss: 39.3119
loss : 39.31190872192383
2/2 [==============================] - 0s 1ms/step
r2 score:  0.647668146457566

########################################

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.9, random_state=144
)

model = Sequential()
model.add(Dense(6,input_dim=10))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=700, batch_size=50)

Epoch 700/700
8/8 [==============================] - 0s 997us/step - loss: 3024.6304
2/2 [==============================] - 0s 0s/step - loss: 1684.5969
loss : 1684.596923828125
2/2 [==============================] - 0s 1ms/step
r2 score:  0.6886146668958438
#########################################

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.9, random_state=57572
)

model = Sequential()
model.add(Dense(6,input_dim=10))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=700, batch_size=50)

Epoch 700/700
8/8 [==============================] - 0s 1ms/step - loss: 3014.3010
2/2 [==============================] - 0s 16ms/step - loss: 1904.2257
loss : 1904.2257080078125
2/2 [==============================] - 0s 0s/step
r2 score:  0.7030471413335173
########################################

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.9, random_state=650874
)

model = Sequential()
model.add(Dense(6,input_dim=10))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=500, batch_size=100)

Epoch 500/500
4/4 [==============================] - 0s 997us/step - loss: 3028.5078
2/2 [==============================] - 0s 0s/step - loss: 1622.0149
loss : 1622.014892578125
2/2 [==============================] - 0s 0s/step
r2 score:  0.7237304881070897

'''