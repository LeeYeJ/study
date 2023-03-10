# 이제 분류할거임
# 이진분류도 다중분류에 속함
# 이전과 다른 부분은 y가 0과 1이라는 것
'''
이진분류때
model.add(Dense(1, activation='sigmoid')) # 마지막 레이어에 sigmoid 주기 #마지막을 고쳐주면됨
model.compile(loss='binary_crossentropy'
'''
#과제=> 파이썬 책에 리스트 딕셔너리 튜플에 대해 공부하고 메일 보내기

import numpy as np
from sklearn.datasets import load_breast_cancer #유방암 데이터 암 걸렸나 안걸렸나
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

#데이터
datasets=load_breast_cancer()
#print(datasets)
print(datasets.DESCR) #DESCR 묘사 #pandas : descibe()
print(datasets.feature_names) # 컬럼이름 , #pandas :.columns()

x=datasets['data']
y=datasets.target

print(x.shape,y.shape) #(569, 30) (569,)
#print(y) # 암t/f

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=333,train_size=0.9
)

model=Sequential()
model.add(Dense(10,activation='relu',input_dim=30))
model.add(Dense(9, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(7, activation='linear'))
model.add(Dense(1, activation='sigmoid')) # 마지막 레이어에 sigmoid 주기 #마지막을 고쳐주면됨 0-1 사이의 수로 나오니까! 밑에서 반올림해줄거임

model.compile(loss='binary_crossentropy', optimizer='adam', #'mse' 실수로 나옴 그러니까 이진분류때는 쓰지말고 binary_crossentropy를 써줌
              metrics=['accuracy','acc','mse', 'r2_score'] #,mean_squared_error #metrics=['accuracy']를 쓰면 알아서 np.round까지 해줌/'mse'도 보고싶으면 메트릭스에서 불러와보면됨,mae도 마찬가지.. 매트릭스놈들은 가능 대신 
              ) # metrics=['accuracy'와 acc=accuracy_score(y_test,y_pre)의 acc와 같은 놈

es = EarlyStopping(monitor='val_accuracy',mode='auto', patience=20,restore_best_weights=True)

model.fit(x_train,y_train, epochs=100, batch_size=8,validation_split=0.1,verbose=1,callbacks=[es])

results=model.evaluate(x_test,y_test)
print('results :', results)
'''
loss,acc,mse 세가지 나옴
results : [0.15466229617595673, 0.9385964870452881, 0.04537041485309601]
'''


# 이분법적이니까 정확도 사용하기 (이진분류)
#회귀 아님 분류
#분류는 이중 아니면 다중(예를 들어 가위,바위,보)밖에 없음
#텐서플로 2번 문제로 분류 나옴

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score #metrics 지표들 들어있음
y_pre=np.round(model.predict(x_test)) # np.round()를 써줘서 예측 값을 반올림 해주자
# print('====================')
# print(y_test[:5])
# print(np.round(y_pre[:5]))  # 반올림은 6부터
# print('====================')

acc=accuracy_score(y_test,y_pre) # accuracy_scores는 y_test,y_pre 둘이 몇프로 맞나
print('acc :',acc)
'''
에러뜨는데
[1 0 1 1 0]
[[2.2693784]
 [1.5962437]
 [1.2626175]
 [2.4894865]
 [2.8863986]]
 비교해줘야 되는데 자료형이 맞지 않음. 그니까 실수를 0과 1로 한정하고싶다. 한정해주자 한정함수
 activation함수에서 0 과 1로 한정할수있는 sigmoid 사용
 Sigmoid 함수는 모든 실수 입력 값을 0보다 크고 1보다 작은 미분 가능한 수로 변환하는 특징을 갖는다.(역전파 가능)
 이진분류는 sigmoid 쓴다!!!!!!!!!!
'''

'''
결과

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=333,train_size=0.9
)

Epoch 37/100
58/58 [==============================] - 0s 1ms/step - loss: 0.2328 - accuracy: 0.9109 - acc: 0.9109 - mse: 0.0673 - val_loss: 0.0574 - val_accuracy: 0.9615 - val_acc: 0.9615 - val_mse: 0.0178
2/2 [==============================] - 0s 2ms/step - loss: 0.1240 - accuracy: 0.9825 - acc: 0.9825 - mse: 0.0189
results : [0.12403340637683868, 0.9824561476707458, 0.9824561476707458, 0.01885361410677433]
acc : 0.9824561403508771 

느낀점 - 같은 조건인데 validation_split를 줄여줬더니 성능이 향상됨
'''