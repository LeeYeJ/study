import numpy as np
from sklearn.svm import LinearSVC, SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

#데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0] # 같으면 0 다르면 1

#모델
# model = LinearSVC()
# model =  SVC()

# 퍼셉트론 겨울 극볶~_!_@
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))

#컴파일 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_data,y_data, batch_size=1, epochs=100)

y_predict = model.predict(x_data) # fit에서 이미 가중치와 바이어스가 생성되었으니까 가능

result = model.evaluate(x_data, y_data)
print('model.score:',result[1])
# result = model.score(x_data,y_data)
# print('model.score :', result) # 모델에서 알아서 분류 모델이니까 acc로 나옴

acc = accuracy_score(y_data,np.round(y_predict))
print('acc :', acc)

#ValueError: Classification metrics can't handle a mix of binary and continuous targets ---> round(y_predict)해결
# y_data -> 넘파이 / y_predict -> 파이썬 오류 .......> np.round()

'''

'''