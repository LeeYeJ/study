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
print(bbb.shape)

x = bbb[:,:-1]
# print(x)
y = bbb[:,-1:]
# print(y)

# def split_x2(x_predict, timesteps):
#     aaa2=[]
#     for i in range(len(x_predict) - timesteps + 1):           =   bbb2 = split_x(x_predict, timesteps=4) 왜냐하면 위에서 이미 정의해줬으니까. 타임스텝스만 바꿔서 쓰면 됨
#         subset = x_predict[i:(i + timesteps)]
#         aaa2.append(subset)
#     return np.array(aaa2)

bbb2 = split_x(x_predict, timesteps=4)
# print(bbb2)
print(bbb2.shape)


###############모델 만들어###############
model= Sequential()
model.add(GRU(16,input_shape=(4,1),activation='linear')) # 행 빼고 나머지/ 다른 모델들도 마찬가지임 / 32는 아웃풋 노드의 갯수
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
x_predict = np.array(bbb2).reshape(7,4,1) 
print(x_predict)

result = model.predict(x_predict)
print('loss : ', loss)
print('[100:107]의 결과값 :', result)
print('걸린 시간은 :', round(end_time-start_time,2))
'''
loss :  0.003750842297449708
[100:107]의 결과값 : [[ 99.98156 ]
 [100.97927 ]
 [101.976906]
 [102.97456 ]
 [103.97217 ]
 [104.969795]
 [105.9674  ]]
걸린 시간은 : 5.77

(Early Stopping 걸어준 결과값)

loss :  0.00012952332326676697
[100:107]의 결과값 : [[100.036674]
 [101.06817 ]
 [102.10021 ]
 [103.13355 ]
 [104.18214 ]
 [105.23133 ]
 [106.28108 ]]
걸린 시간은 : 24.54

GRU 써줌
loss :  8.783402881817892e-05
[100:107]의 결과값 : [[100.02781 ]
 [101.030075]
 [102.032455]
 [103.034935]
 [104.03753 ]
 [105.04022 ]
 [106.04302 ]]
걸린 시간은 : 101.87
'''






