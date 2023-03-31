# 시계열은 데이터를 어떻게 자르느냐가 중요하다
#https://www.w3schools.com/python/numpy/numpy_array_slicing.asp 참고
# 넘파이 어레이 슬라이싱

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping

dataset = np.array(range(1,101))
timesteps =5 # 5개씩 잘라
x_predict = np.array(range(96,106)) # 100 - 106 예상값

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i:(i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(dataset, timesteps)
print(bbb)

x = bbb[:,:-1]
print(x)
y = bbb[:,-1:]
print(y)

# def split_x2(x_predict, timesteps):
#     aaa2=[]
#     for i in range(len(x_predict) - timesteps + 1):           =   bbb2 = split_x(x_predict, timesteps=4) 왜냐하면 위에서 이미 정의해줬으니까. 타임스텝스만 바꿔서 쓰면 됨
#         subset = x_predict[i:(i + timesteps)]
#         aaa2.append(subset)
#     return np.array(aaa2)

bbb2 = split_x(x_predict, timesteps=4)
print(bbb2)


###############모델 만들어###############
model= Sequential()
model.add(Dense(16,input_shape=(4,),activation='linear')) # 행 빼고 나머지/ 다른 모델들도 마찬가지임 / 32는 아웃풋 노드의 갯수
model.add(Dense(16,activation='relu'))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(10,activation='relu'))
model.add(Dense(16))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
import time
start_time = time.time()
es=EarlyStopping(monitor='val_loss',mode='auto',patience=20,restore_best_weights=True)
model.fit(x,y,epochs=2000,callbacks=[es])
end_time= time.time()

#4.평가예측
loss = model.evaluate(x,y)
x_predict = np.array(bbb2) 
print(x_predict.shape)

result = model.predict(x_predict)
print('loss : ', loss)
print('[100:107]의 결과값 :', result)
print('걸린 시간은 :', round(end_time-start_time,2))