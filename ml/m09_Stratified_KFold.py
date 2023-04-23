# 분류 문제에서만 씀
# 라벨을 균등히 분포

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_val_predict,train_test_split,KFold,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

x,y = load_iris(return_X_y=True)

x_train,x_test, y_train,y_test = train_test_split(
    x,y , shuffle=True, random_state=337, test_size=0.2
)

n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True,random_state=337)
kf = StratifiedKFold(n_splits=n_splits, shuffle=True,random_state=337)

model=SVC()
# model =RandomForestClassifier()

#컴파일 훈련 평가 예측
score = cross_val_score(model,x_train,y_train,cv =kf) 
print('cross_val_score :',score,'\n 교차검증평균점수:',
      round(np.mean(score),4)) # 훈련데이터로만 함

y_predict = cross_val_predict(model,x_test,y_test,cv =kf)
print('cross_val_predict ACC :',accuracy_score(y_test,y_predict))

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))
'''
cross_val_predict ACC : 0.8333333333333334 비포
cross_val_predict ACC : 0.9666666666666667 에프터

편향 데이터 판단은 f1 스코어로 많이 해준다.

'''
