import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold


#1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']

X,y = load_iris(return_X_y=True)
print(X.shape, y.shape)

#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# 트리 구조의 모델들은 결측치와 이상치로부터 자유롭다. 
from sklearn.tree import DecisionTreeClassifier # 분류면 Classfier 써주면 됨 / 실수나 연속수는 Regression 사용해주면 됨
# Decision이 여러개 있으면 랜덤 포레스트 (like 나무와 숲)

from sklearn.ensemble import RandomForestRegressor
n_splits = 5 # 디폴트값 5
kfold = KFold(n_splits = n_splits, shuffle=True,random_state=123) #

# model = LinearSVC() # 파라미터 c가 클수록 직선이다. 작으면 더 정교하게 데이터의 영역을 나눠준다. 
# model - LogisticRegression()
model = DecisionTreeClassifier()

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

scores = cross_val_score(model,X,y,cv=5, n_jobs=-1) # cv = 5라고 써도 됨 / 위에서 정의해줘도 되고 /n_jobs=-1 최대 쓰는거임
print('ACC :',scores,'\n cross_val_score 평균 :',round(np.mean(scores),4))

# model.fit(X,y) # fit에 컴파일 포함되어있음

#4.평가예측
# results = model.evaluate(x,y)
# print(results)

# result = model.score(X,y)
# print(result) 
'''
-KFOLD-
ACC : [0.96666667 0.96666667 0.9        0.96666667 1.        ] 
 cross_val_score 평균 : 0.96
'''

