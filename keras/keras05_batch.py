import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])


#2. 모델 구성 
model = Sequential()
model.add(Dense(5,input_dim=1)) 
model.add(Dense(6)) #튜닝
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일 , 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=495, batch_size=1)  # batch_size 디폴트 값은 32 -> 설명은 탭 노트에 해둠  

# 4. 평가, 예측
loss = model.evaluate(x,y) # 위에 가중치에 데이터를 넣어서 무슨 값이 나오는지 평가한다. (나중에는 데이터 안넣음 왜냐 훈련되지않은 데이터를 평가하기 위해) 즉, 모델의 정확도를 평가힐 수 있다.git 
print('loss : ', loss)

result = model.predict([6])
print('[6]의 예측값은 : ', result)
