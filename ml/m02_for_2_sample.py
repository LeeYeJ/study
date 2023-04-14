#분류 데이터들 싹 모아서 해봐

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer 
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터

data_list=[load_iris(return_X_y=True)
        ,load_iris(return_X_y=True)
        ,load_breast_cancer(return_X_y=True)
]  # 함수나 여기서 사용가능

# datasets = load_iris()
# x = datasets.data
# y = datasets['target']

# x,y = load_iris(return_X_y=True)
# x,y = load_wine(return_X_y=True)
# x,y = load_digits(return_X_y=True)

#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# 트리 구조의 모델들은 결측치와 이상치로부터 자유롭다. 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # 분류면 Classfier 써주면 됨 / 실수나 연속수는 Regression 사용해주면 됨
# Decision이 여러개 있으면 랜덤 포레스트 (like 나무와 숲)

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

# model = LinearSVC() # 파라미터 c가 클수록 직선이다. 작으면 더 정교하게 데이터의 영역을 나눠준다. 
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = DecisionTreeRegressor()

model_list=[LinearSVC(),
            LogisticRegression(),
            DecisionTreeClassifier(),
            RandomForestClassifier()]

data_name_list =['아이리스 :',
                 '브레스트캔서 :',
                 '와인 :']

model_name_list = ['LinearSVC :',
                   'LogisticRegression :',
                   'DecissionClaafier :',
                   'RF :'
]


# model =  Sequential()
# model.add(Dense(10,activation='relu',input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

#3.컴파일 훈련
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc']) # 위에서 원핫하지 않은 경우엔 sparse_categorical_crossentropy로 원핫 됨

for i , value in enumerate(data_list):
    x,y = value
    # print(x.shape, y.shape)
    print('===============================')
    print(data_name_list[i])
    
    for j , value2 in enumerate(model_list):
        model = value2
        # 컴파일 훈련
        model.fit(x,y)
        # 평가 예측
        result = model.score(x,y)
        print(model_name_list[j],'model.score :',result)
        y_pred = model.predict(x)
        acc = accuracy_score(y,y_pred)
        print(model_name_list[j],'acc :', acc)

'''
===============================
아이리스 :
LinearSVC : model.score : 0.9666666666666667
LinearSVC : acc : 0.9666666666666667
LogisticRegression : model.score : 0.9733333333333334
LogisticRegression : acc : 0.9733333333333334
DecissionClaafier : model.score : 1.0
DecissionClaafier : acc : 1.0
RF : model.score : 1.0
RF : acc : 1.0
===============================
브레스트캔서 :
LinearSVC : model.score : 0.9666666666666667
LinearSVC : acc : 0.9666666666666667
LogisticRegression : model.score : 0.9733333333333334
LogisticRegression : acc : 0.9733333333333334
DecissionClaafier : model.score : 1.0
DecissionClaafier : acc : 1.0
RF : model.score : 1.0
RF : acc : 1.0
===============================
와인 :
LinearSVC : model.score : 0.9332161687170475
LinearSVC : acc : 0.9332161687170475
LogisticRegression : model.score : 0.9472759226713533
LogisticRegression : acc : 0.9472759226713533
DecissionClaafier : model.score : 1.0
DecissionClaafier : acc : 1.0
RF : model.score : 1.0
RF : acc : 1.0
'''

# model.fit(x,y) # fit에 컴파일 포함되어있음

#4.평가예측
# results = model.evaluate(x,y)
# print(results)

# result = model.score(x,y)
# print(result) 

'''
iris 
1.0.9666666666666667
2.0.9733333333333334
3.1.0
4.1.0

wine
1.0.9213483146067416
2.0.9662921348314607
3.1.0
4.1.0

digits
1.0.9838619922092376
2.1.0
3.1.0
4.1.0

fetch
1.
2.0.618729045183232
3.1.0
4.1.0


'''