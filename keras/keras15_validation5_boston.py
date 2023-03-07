#핏에서 validation_split=0.2로 검증해줌

#보스턴에 있는 집값을 찾는 지표
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

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
model.fit(x_train,y_train,epochs=3000, batch_size=100 , validation_split=0.2)

#4.평가 예측
loss=model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict=model.predict(x_test) # 훈련 안시킨 데이터에서 예측하자 아래

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # 원값과 예측값이 얼마나 맞는지 확인할 수 있다. / 얘도 훈련안한 y_test로 확인해보자 (내신,수능 비교)
print('r2스코어 :', r2) # 값은 1과 가까울 수록 좋다.

'''
Epoch 2999/3000
3/3 [==============================] - 0s 11ms/step - loss: 25.1563 - val_loss: 34.1652
Epoch 3000/3000
3/3 [==============================] - 0s 10ms/step - loss: 25.2835 - val_loss: 35.3203
5/5 [==============================] - 0s 836us/step - loss: 20.0264
loss :  20.026399612426758
5/5 [==============================] - 0s 743us/step
r2스코어 : 0.7947525491471229
'''


