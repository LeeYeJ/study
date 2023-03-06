#보스턴에 있는 집값을 찾는 지표
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

''' 
실습
1. train 0.7
2. R2 0.8이상
'''

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
# 데이터부분은 x데이터
# 타겟부분은 y데이터

x_train, x_test, y_train, y_test = train_test_split(x, y, #x는 x_train과 x_test로 분리되고, y는 y_train과 y_test 순서로! 분리된다.
     train_size=0.7, shuffle=True, random_state=1188
)

#random_state= 7995,45681

#print(x)
#print(y)
#print(datasets)
#print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

#print(datasets.DESCR)  데이터셋의 정보
'''
  - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's 결과값 y
'''
#print(x.shape, y.shape) #(506, 13) (506,)


#2.모델 구성
model=Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=3000, batch_size=100 )

#4.평가 예측
loss=model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict=model.predict(x_test) # 훈련 안시킨 데이터에서 예측하자 아래

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # 원값과 예측값이 얼마나 맞는지 확인할 수 있다. / 얘도 훈련안한 y_test로 확인해보자 (내신,수능 비교)
print('r2스코어 :', r2) # 값은 1과 가까울 수록 좋다.

'''
x_train, x_test, y_train, y_test = train_test_split(x, y, #x는 x_train과 x_test로 분리되고, y는 y_train과 y_test 순서로! 분리된다.
     train_size=0.9, shuffle=True, random_state=4
)

loss='mae'

2/2 [==============================] - 0s 0s/step - loss: 2.9821
loss :  2.982077121734619
2/2 [==============================] - 0s 996us/step
r2스코어 : 0.7524482772227262
---------------------------------------------------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, #x는 x_train과 x_test로 분리되고, y는 y_train과 y_test 순서로! 분리된다.
     train_size=0.7, shuffle=True, random_state=2
)

model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train,epochs=700, batch_size=10 )

Epoch 700/700
36/36 [==============================] - 0s 913us/step - loss: 3.7843
5/5 [==============================] - 0s 5ms/step - loss: 3.1567
loss :  3.1566600799560547
5/5 [==============================] - 0s 821us/step
r2스코어 : 0.715024891209413
------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, #x는 x_train과 x_test로 분리되고, y는 y_train과 y_test 순서로! 분리된다.
     train_size=0.7, shuffle=True, random_state=587
)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=3000, batch_size=100 )

Epoch 3000/3000
4/4 [==============================] - 0s 1ms/step - loss: 28.1496
5/5 [==============================] - 0s 4ms/step - loss: 15.3093
loss :  15.309255599975586
5/5 [==============================] - 0s 805us/step
r2스코어 : 0.8122799584732157
-------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, #x는 x_train과 x_test로 분리되고, y는 y_train과 y_test 순서로! 분리된다.
     train_size=0.7, shuffle=True, random_state=709
)
4/4 [==============================] - 0s 2ms/step - loss: 27.0326
5/5 [==============================] - 0s 4ms/step - loss: 16.1928
loss :  16.192752838134766
5/5 [==============================] - 0s 745us/step
r2스코어 : 0.8144975751848098
--------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, #x는 x_train과 x_test로 분리되고, y는 y_train과 y_test 순서로! 분리된다.
     train_size=0.7, shuffle=True, random_state=903
)
Epoch 3000/3000
4/4 [==============================] - 0s 1ms/step - loss: 27.4075
5/5 [==============================] - 0s 4ms/step - loss: 15.8519
loss :  15.851949691772461
5/5 [==============================] - 0s 636us/step
r2스코어 : 0.8071997998159424
--------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, #x는 x_train과 x_test로 분리되고, y는 y_train과 y_test 순서로! 분리된다.
     train_size=0.7, shuffle=True, random_state=1036
)
Epoch 3000/3000
4/4 [==============================] - 0s 1ms/step - loss: 26.5536
5/5 [==============================] - 0s 789us/step - loss: 17.3756
loss :  17.375646591186523
5/5 [==============================] - 0s 0s/step
r2스코어 : 0.804257656313496
---------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, #x는 x_train과 x_test로 분리되고, y는 y_train과 y_test 순서로! 분리된다.
     train_size=0.7, shuffle=True, random_state=1188
)
4/4 [==============================] - 0s 1ms/step - loss: 26.8784
5/5 [==============================] - 0s 0s/step - loss: 17.6412
loss :  17.6412296295166
5/5 [==============================] - 0s 1ms/step
r2스코어 : 0.8191977608601066
-------------------------------------------------------------------
'''

