import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.데이터 (+분리)
x= np.array([[1,2,3,4,5],[6,7,8,9,10]])

y=np.array([11,12,13,14,15])

#전치
x=x.T 

#print(x.shape) #(5,2)

#데이터 분리
x_test,x_train,y_test,y_train= train_test_split(
    x,y,
    shuffle=True,
    train_size=0.6, # 만약 6:3으로 나누면 10프로는 날아감.
    random_state=1 # 랜덤 데이터 고정값
    
)


#2.모델 구성
model=Sequential()
model.add(Dense(5,input_dim=2))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs=500, batch_size=1)

#4.평가 및 훈련
loss=model.evaluate(x_test,y_test)
print('loss :', loss)

result=model.predict([[6,11]])
print('[[6,11]]의 result :', result)


'''
model=Sequential()
model.add(Dense(5,input_dim=2))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs=500, batch_size=1)

Epoch 500/500
2/2 [==============================] - 0s 2ms/step - loss: 0.0113
1/1 [==============================] - 0s 96ms/step - loss: 0.0089
loss : 0.008913676254451275
1/1 [==============================] - 0s 79ms/step
[[6,11]]의 result : [[16.009937]]
'''