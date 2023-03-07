#슬라이싱은 행기준 자름
#슬라이싱을 해서 val 변수를 만들어서 fit에 val 데이터 넣어줌

from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_diabetes()
x = datasets.data
y = datasets.target

#print(x.shape, y.shape) #(442, 10) (442,)
x_train=x[:400]
y_train=y[:400]

x_test=x[399:419]
y_test=y[399:419]

x_val=x[420:]
y_val=y[420:]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(x_val.shape)


'''
[실습]
R2 0.62 이상
'''


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
model.fit(x_train,y_train, epochs=500, batch_size=100,validation_data=(x_val,y_val) )

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

r2= r2_score(y_test, y_predict) #loss와 같이 판단해야한다, r2는 보조지표이기때문에!
print(x_test.shape,y_test.shape,y_predict.shape)
print('r2 score: ', r2)
'''
Epoch 499/500
4/4 [==============================] - 0s 7ms/step - loss: 3011.2620 - val_loss: 1850.8898
Epoch 500/500
4/4 [==============================] - 0s 7ms/step - loss: 3011.1370 - val_loss: 1843.2203
1/1 [==============================] - 0s 15ms/step - loss: 1560.3230
loss : 1560.322998046875
(20, 10) (20,) (20, 1)
r2 score:  0.6955139847953793
'''