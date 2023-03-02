import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10]) 

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법! / 사이킷런
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.3, 
    #train_size=0.7,
    random_state=1234, # 랜덤값을 고정한다!! 랜덤씨드로 고정값! 또는 랜덤 스테이트 123에 있는 형식대로
    shuffle=True, # 디폴트가 트루
    )

print(y_train)
print(y_test)

#2.모델 구성
model = Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=500,batch_size=1)

loss= model.evaluate(x_test,y_test)
print('loss :', loss)

result=model.predict([11])
print('result :', result)

'''
Epoch 500/500
7/7 [==============================] - 0s 906us/step - loss: 2.0961e-13
1/1 [==============================] - 0s 111ms/step - loss: 1.7053e-13
loss : 1.7053025658242404e-13
1/1 [==============================] - 0s 69ms/step
result : [[11.000001]]
''' 
'''
 random_state=1234, # 랜덤값을 고정한다!! 그래야 평가가 의미가 있으니까. 랜덤씨드로 고정값! 또는 랜덤 스테이트 123에 있는 형식대로
[2 1 9 5 6 7 4]
[ 8  3 10]

[2 1 9 5 6 7 4]
[ 8  3 10]

즉 8,3,10 고정
'''




