import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([range(10), range(21,31), range(201,211)])  # range함수는 크기  range(10) -> [0,1,2,3,4,5,6,7,8,9]
x = x.T #(10,3)
#print(x)
#print(x.shape) #(3,10)

y= np.array([[1,2,3,4,5,6,7,8,9,10]])
y=y.T #(10,1)

#모델구성
model = Sequential()
model.add(Dense(3,input_dim=3)) # x의 열 갯수가 인풋딤이 됨
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1)) # 전치해서 1열이 됐음으로 츨력 1가능 즉, y 열의 갯수가 아웃풋값이 됨

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=500,batch_size=1)

#4.평가 예측
loss = model.evaluate(x,y,batch_size=1)
print('loss :' , loss)

result = model.predict([[9,30,210]])
print('result :', result)
'''
Epoch 500/500
10/10 [==============================] - 0s 881us/step - loss: 4.2036e-11
1/1 [==============================] - 0s 98ms/step - loss: 4.5719e-11
loss : 4.5719161095858496e-11          # 이 평가부분의 loss는 전체 트레이닝 과정의 전체 로스값의 평균값이 들어있는 것이다.
1/1 [==============================] - 0s 68ms/step
result : [[9.999993]]
'''

# 궁금한 점 : 위 결과값이 근사친데 loss는 4..?가 나올 수 있는 것인가 e-11..?

# 이 시점에서 확실히 알아보고 가고 싶은 부분 -> 평가하는 부분. 아래 설명

'''
evaluate 함수에 데이터를 넣으면 두 가지 결과를 보여주는데 첫 번째는 바로 오차값(loss)입니다. 
오차값은 0~1 사이의 값으로, 0이면 오차가 없는 것이고 1이면 오차가 아주 크다는 것을 의미합니다. 
두 번째는 정확도(accuracy)입니다. 모델이 예측한 값과 정답이 얼마나 정확한지에 대해서 0과 1 사이의 값으로 보여줍니다. 
1에 가까울 수록 정답을 많이 맞춘 것을 의미합니다.
'''