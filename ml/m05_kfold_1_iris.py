# 교차 검증

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1.데이터
x,y = load_iris(return_X_y=True)

# x_train,x_test,y_train,y_test = train_test_split(
#     x,y, shuffle=True, random_state=123, test_size=0.2
# )

n_splits = 5 # 디폴트값 5
kfold = KFold(n_splits = n_splits, shuffle=True,random_state=123) # cross_val할 내용 정의 부분 /n_splits 나누기
# kfold = KFold() 디폴트 있음
 

#2모델
model = LinearSVC()

#3 4 . 컴파일 훈련 평가 예측
# scores = cross_val_score(model,x,y,cv=kfold) # 모델 / 데이터 / 크로스발을 어떻게 할것인지
scores = cross_val_score(model,x,y,cv=5, n_jobs=-1) # cv = 5라고 써도 됨 / 위에서 정의해줘도 되고 /n_jobs=-1 최대 쓰는거임

print(scores)

print('ACC :',scores,'\n cross_val_score 평균 :',round(np.mean(scores),4))

'''
[0.96666667 1.         0.93333333 0.93333333 0.9       ] -> kfold 갯수만큼 훈련

'''