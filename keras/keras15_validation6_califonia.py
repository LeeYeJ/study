#train_test_split로 나눠줘서 핏에서 검증함

from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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
x_train,x_val,y_train,y_val=train_test_split(
    x_train,y_train,train_size=0.5,random_state=123
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
model.fit(x_train,y_train, epochs=700, batch_size=50, validation_data=(x_val,y_val))

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

r2= r2_score(y_test, y_predict)
print('r2 score: ', r2)

'''
x_train,x_val,y_train,y_val=train_test_split(
    x_train,y_train,train_size=0.5,random_state=123
)  
val 데이터를 나누어 주고
validation_data=(x_val,y_val)
fit에서 발리할 각 변수를 넣어준다.

여기까지가 한 세트
'''