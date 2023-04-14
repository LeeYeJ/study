import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,1] # 1이 하나라도 있으면 1

#모델
model = LinearSVC()

model.fit(x_data,y_data)

y_predict = model.predict(x_data) # fit에서 이미 가중치와 바이어스가 생성되었으니까 가능

result = model.score(x_data,y_data)
print('model.score :', result) # 모델에서 알아서 분류 모델이니까 acc로 나옴

acc = accuracy_score(y_data,y_predict)
print('acc :', acc)
'''
model.score : 1.0
acc : 1.0
'''