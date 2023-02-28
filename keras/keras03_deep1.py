# 1. 데이터
import numpy as np # numpy의 행렬이 사람이 하는 것과 유사하기 때문에 자주 사용
x = np.array([1,2,3]) #[] 한 덩어리로 봄
y = np.array([1,2,3]) #마찬가지로 한 덩어리로 들어감, 각각의 데이터는 그래프로 대각선 세 점의 데이터

# 2. 모델 구성
import tensorflow as tf
from tensorflow.keras.models import Sequential # 시퀀셜을 쓴 이유는 순차적 모델 클래스를 가져다쓰기 때문. 모델은 총 두가지 함수형 / 시퀀셜.
from tensorflow.keras.layers import Dense # 각각의 노드 층이 존재하고 단순한 연산일 경우 (y=wx+b) Dense 사용

model = Sequential() # 시퀀셜을 가져다 모델이라 해준다. / model이름은 나의 자유 ex) model4
model.add(Dense(3, input_dim=1)) # Dense로 층으로 쌓는다 add , 3은 아웃풋 , input_dim은 x데이터 [] 한 덩어리로 들어감.
model.add(Dense(4)) # 상위 그대로가 인풋이니까 명시 안해도 됨
model.add(Dense(5))
model.add(Dense(3)) # 3은 하이퍼 파라미터 위도 마찬가지
model.add(Dense(1)) 

# ctrl + / 키는 줄 자체를 주석으로 바꿔줌
# shift + delete / 줄 삭제
# 줄에 커서 놓고 ctrl+ c / 라인 카피


# 3. 컴파일, 훈련 / 위에 모델 정의만 해놓았기 때문에 
model.compile(loss='mse', optimizer='adam') # mse는 차이 연산, 그것에 대한 최적화는 adam을 쓴다.
model.fit(x,y, epochs=100)  # 훈련 각각의 데이터 값을 널어주고 반복 훈련

# loss: 0.0067