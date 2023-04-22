# 회귀데이터 싹 모아서 해봥

import numpy as np
from sklearn.datasets import load_boston,load_diabetes,fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold,cross_val_predict
from sklearn.metrics import r2_score

x,y = load_boston(return_X_y=True)


x_train,x_test,y_train,y_test=train_test_split(
    x,y, shuffle=True, random_state=1234
)

#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression # 시그모이드라고 생각해
# 트리 구조의 모델들은 결측치와 이상치로부터 자유롭다. 
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor# 분류면 Classfier 써주면 됨 / 실수나 연속수는 Regression 사용해주면 됨
# Decision이 여러개 있으면 랜덤 포레스트 (like 나무와 숲)

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

model = RandomForestRegressor()

scores = cross_val_score(model,x_train,y_train,cv=5, n_jobs=-1) # cv = 5라고 써도 됨 / 위에서 정의해줘도 되고 /n_jobs=-1 최대 쓰는거임
y_pred = cross_val_predict(model,x_test,y_test,cv =kfold)

r2 = r2_score(y_test,y_pred)
print('r2_score :',scores,'\n cross_val_score 평균 :',round(np.mean(scores),4))
print('cross_val_predict_R2 :',r2)
'''
 cross_val_score 평균 : 0.6094
 ========================================================
  cross_val_score 평균 : 0.8147
cross_val_predict_R2 : 0.7907161571612892
'''

##########에러############
# model - LogisticRegression() # Regression이지만 분류모델임 -> 얘도 에러뜸
# model = RandomForestClaasfier() # 에러뜸 분류문제가 아닌데 분류 모델을 써줬으니까
# model = DecisionTreeClassifier()
# model = LinearSVC() # 파라미터 c가 클수록 직선이다. 작으면 더 정교하게 데이터의 영역을 나눠준다. 



# model =  Sequential()
# model.add(Dense(10,activation='relu',input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))


# models = {
#     'DecisionTreeRegressor': DecisionTreeRegressor(),
#     'RandomForestRegressor': RandomForestRegressor()
# }

# #3.컴파일 훈련
# # model.compile(loss='sparse_categorical_crossentropy',
# #               optimizer='adam',
# #               metrics=['acc']) # 위에서 원핫하지 않은 경우엔 sparse_categorical_crossentropy로 원핫 됨

# from sklearn.model_selection import train_test_split
# from tensorflow.keras.metrics import mean_squared_error,Accuracy


# for dataset_name, dataset in datasets.items():  #.item() - 딕셔너리의 키(key)와 값(value)을 모두 반환
#     X, y = dataset
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f'\n===== {dataset_name} dataset =====')

#     for model_name, model in models.items():
#         print(f"\n[ {model_name} ]")
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         mse = mean_squared_error(y_test, y_pred)
#         print(f"Mean squared error: {mse:.4f}")
    
# model.fit(X,y) # fit에 컴파일 포함되어있음

#4.평가예측
# results = model.evaluate(x,y)
# print(results)

# result = model.score(X,y)
# print(result) 

'''
boston
1.1.0
2.0.9832829628350938

california
1.1.0
2.0.917859537384686

diabets
1.1.0
2.0.9740496467598819


'''