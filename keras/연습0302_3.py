#다시 해보기

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.데이터
x= np.array([[1,2,3],[6,7,8],[11,12,13]])
y= np.array([16,17,18])  # --> 궁금한점. 차원을 늘렸을때 왜 안됨..? 다시 해보귀...



x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    shuffle=True,
    train_size=0.7,
    random_state=1
)



'''
#print(x_train)

'''
model=Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=500,batch_size=1)

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

result= model.predict([[3,5,7]])
print('result :', result)
