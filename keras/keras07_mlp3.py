import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array( #행무시, 열우선
    [
        [1,2,3,4,5,6,7,8,9,10],
        [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
        [9,8,7,6,5,4,3,2,1,0]
    ]   
)
y = np.array([11,12,13,14,15,16,17,18,19,20])

# 전치 실습
x = x.transpose()

print(x.shape)
print(y.shape)

# 예측할 값은 [[10,1.4,0]]

model = Sequential()
model.add(Dense(3,input_dim=3))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=500,batch_size=1)

loss = model.evaluate(x,y)
print('loss :' , loss)

result = model.predict([[10,1.4,0]])
print('[[10,1.4,0]]의 result :', result)

'''
loss : 0.0026700710877776146
1/1 [==============================] - 0s 85ms/step
result : [[20.01697]]
model = Sequential()
model.add(Dense(3,input_dim=3))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=500,batch_size=1)

loss = model.evaluate(x,y)
print('loss :' , loss)

result = model.predict([[10,1.4,0]])
print('result :', result)

디버그 페이지
Epoch 498/500
10/10 [==============================] - 0s 816us/step - loss: 0.0042
Epoch 499/500
10/10 [==============================] - 0s 995us/step - loss: 0.0042
Epoch 500/500
10/10 [==============================] - 0s 891us/step - loss: 0.0045   # 10/10 batch_size
1/1 [==============================] - 0s 87ms/step - loss: 0.0027  # 평가
loss : 0.0026700710877776146
1/1 [==============================] - 0s 85ms/step  # 예측
result : [[20.01697]]

'''

