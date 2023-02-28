import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[4,3,5],[2,5,6]])
y = np.array([1,3])

#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=3)) # x의 열이 3
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

#4.평가 예측
loss = model.evaluate(x,y)
print("loss :", loss)

result = model.predict([[4,3,5]]) # x가 2차원이니까 출력도 마찬가지 [[]]
print("result :", result)

# 2월 28일 batch  , 행렬 , 입력의 컬럼 수가 중요함! 이로인해 예측값을 얻을때도 같은 차원의 값이어야 출력이 됨.

#loss : 0.0
#1/1 [==============================] - 0s 75ms/step
#result : [[1.]]

