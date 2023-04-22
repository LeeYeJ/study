import numpy as np
from sklearn.datasets import load_digits,load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,HalvingGridSearchCV
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import time
import pandas as pd

# 1. 데이터
x,y = load_wine(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, # stratify=y 사용함으로써 각 라벨값이 골고루 분배된다 
)    

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True,random_state=1234)


parameters = [
    {'n_estimators':[100,200]},
    {'max_depth':[6,8,10,12]},
    {'min_sample_leaf':[3,4,7,10]},
    {'min_samples_split':[2,3,5,10]},
    {'n_jobs':[-1,2,4]}
]

# 파라미터 조합으로 2개 이상 엮을것

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12],'min_samples_leaf':[3,10]},
    {'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10],'min_samples_split':[2,3,5,10]},
    {'min_samples_split':[2,3,5,10],'max_depth':[6,8,10,12]}
]
#2.모델
model = HalvingGridSearchCV(RandomForestClassifier(), # GridSearchCV에서 랜덤하게 빼서 쓰는 것
                     parameters, 
                     cv = 5,  # 분류의 디폴트는 StratifiedKFold이다.
                    #  cv = kf,  
                     verbose=1, 
                     factor =3.2, # 실수형도 가능
                    #  refit=True, # 최적의 값을 보관함 / 최적의 값을 출력 -> 통상적으로 True로 함
                    #  refit=False, # 모델이 돌아갈때 최적값을 저장하지 않음 -> False하면 최종 파라미터로 출력
                     n_jobs=-1) # GridSearchCV 안에 사용할 모델과 파라미터 정의


#3.컴파일 훈련
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()

print('걸린 시간 :',round(end_time - start_time,2),'초')

print('최적의 매개변수 :',model.best_estimator_) # 가장 좋은 평가 뽑기

print('최적의 파라미터 :',model.best_params_) # 가장 좋은 파람 뽑기

print('best_score_ :',model.best_score_) # 가장 좋은 점수

print('model.score :',model.score(x_test,y_test)) # 테스트한 모델 스코어 (중요)

y_predict = model.predict(x_test)
print('acc :',accuracy_score(y_test,y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :',accuracy_score(y_test,y_pred_best))


print('걸린 시간 :',round(end_time - start_time,2),'초')

'''
Fitting 5 folds for each of 19 candidates, totalling 95 fits
걸린 시간 : 19.02 초
최적의 매개변수 : RandomForestClassifier(max_depth=6, min_samples_leaf=5)
최적의 파라미터 : {'max_depth': 6, 'min_samples_leaf': 5}
best_score_ : 0.9678362573099415
model.score : 0.9722222222222222
acc : 0.9722222222222222
최적 튠 ACC : 0.9722222222222222
걸린 시간 : 19.02 초
'''



