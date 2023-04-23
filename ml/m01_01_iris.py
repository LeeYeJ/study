#분류 데이터들 싹 모아서 해봐

# 성능 지표
# 회귀 r2
# 분류 acc

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, fetch_covtype
from sklearn.model_selection import train_test_split, cross_val_score, KFold


#1. 데이터
x,y = load_iris(return_X_y=True)

n_splits = 5 # 디폴트값 5
kfold = KFold(n_splits = n_splits, shuffle=True,random_state=123) #
# x = datasets.data
# y = datasets['target']

# x,y = load_iris(return_X_y=True)
# x,y = load_wine(return_X_y=True)
# x,y = load_digits(return_X_y=True)
# x,y = fetch_covtype(return_X_y=True)

# print(x.shape, y.shape)
#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# 트리 구조의 모델들은 결측치와 이상치로부터 자유롭다.    
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # 분류면 Classfier 써주면 됨 / 실수나 연속수는 Regression 사용해주면 됨
# Decision이 여러개 있으면 랜덤 포레스트 (like 나무와 숲)

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

# model_list = [LinearSVC(),
#               LogisticRegression(),
#               DecisionTreeClassifier(),
#               DecisionTreeRegressor()]

# model_name_list = ['LinearSVC',
#                    'LogisticRegression',
#                    'DecisionTreeClassifier',
#                    'DecisionTreeRegressor']
from sklearn.model_selection import train_test_split

model = RandomForestClassifier()

scores = cross_val_score(model,x,y,cv=5, n_jobs=-1) # cv = 5라고 써도 됨 / 위에서 정의해줘도 되고 /n_jobs=-1 최대 쓰는거임
print('ACC :',scores,'\n cross_val_score 평균 :',round(np.mean(scores),4))

# print(scores)

'''
ACC : [0.96666667 0.96666667 0.93333333 0.96666667 1.        ]
 cross_val_score 평균 : 0.9667
'''


# 반복문!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# for i , value in enumerate(data_list):
#     x, y = value(return_X_y=True)
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=333)
    
#     # print(x.shape, y.shape)
#     print('===============================')
#     print(data_name_list[i])
    
#     for j , value2 in enumerate(model_list):
#         model = value2
#         model.fit(x_train,y_train)
#         # y_pred = model.predict(x_test)
#         result = model.score(x_test,y_test)
#         print(model_name_list[j],result)

# model = LinearSVC() # 파라미터 c가 클수록 직선이다. 작으면 더 정교하게 데이터의 영역을 나눠준다. 
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = DecisionTreeRegressor()


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