# 1. 데이터
import numpy as np 
x = np.array([1,2,3]) 
y = np.array([1,2,3]) 

# 2. 모델 구성
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

model = Sequential() 
model.add(Dense(3, input_dim=1)) 
model.add(Dense(4)) 
model.add(Dense(5))
model.add(Dense(3)) 
model.add(Dense(1)) 



# 3. 컴파일, 훈련  / 이 단계에서 가중치 생성
model.compile(loss='mse', optimizer='adam') 
model.fit(x,y, epochs=560)  # w값이 계속 갱신

# 4. 평가, 예측
loss = model.evaluate(x,y) # 위에 가중치에 데이터를 넣어서 무슨 값이 나오는지 평가한다. (나중에는 데이터 안넣음 왜냐 훈련되지않은 데이터를 평가하기 위해) 즉, 모델의 정확도를 평가힐 수 있다.git 
print('loss : ', loss)

result = model.predict([4]) # 예측하는 값 4
print('[4]의 예측값은 : ', result) # 예측하는 건 웨이트 값과 예측할 값을 곱해준다

# [4]의 예측값은 :  [[3.9997404]] --> epochs = 560 변경
# loss :  1.665091531322105e-08