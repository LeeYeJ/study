# factor크기만큼 엔빵해서 많은 훈련
# HalvingGridSearchCV에서 n_iter 매개변수 못씀!

import numpy as np
from sklearn.datasets import load_digits,load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,HalvingGridSearchCV
import time
import pandas as pd

# 1. 데이터
x,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, # stratify=y 사용함으로써 각 라벨값이 골고루 분배된다 
)    

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True,random_state=1234)

parameters = [
    {'C':[1,10,100,1000],'kernel':['linear'],'degree':[3,4,5]}, # 12번 돌음
    {'C':[1,10,100],'kernel':['rbf','linear'],'gamma':[0.001,0.0001]}, # 12번 돌음
    {'C':[1,10,100,1000],'kernel':['sigmoid'],
     'gamma':[0.01,0.001,0.0001],'degree':[3,4]}, # 24번 돌음
    {'C':[0.1,1],'gamma':[1,10]} # 4번 돌음
]          # 총 52번 돌음

#2.모델
model = HalvingGridSearchCV(SVC(), # GridSearchCV에서 랜덤하게 빼서 쓰는 것
                     parameters, 
                     cv = 5,  # 분류의 디폴트는 StratifiedKFold이다.
                    #  cv = kf,  
                     verbose=1, 
                    #  factor =3.2, # 실수형도 가능
                    #  n_iter=10, # 디폴트는 10, 10* cv 만큼 훈련 -> 반복 횟수
                    #  refit=True, # 최적의 값을 보관함 / 최적의 값을 출력 -> 통상적으로 True로 함
                    #  refit=False, # 모델이 돌아갈때 최적값을 저장하지 않음 -> False하면 최종 파라미터로 출력
                     n_jobs=-1) # GridSearchCV 안에 사용할 모델과 파라미터 정의


#3.컴파일 훈련
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()

print('걸린 시간 :',round(end_time - start_time,2),'초')
# 걸린 시간 : 3.18 초

# print(x.shape,x_train.shape) # (1797, 64) (1437, 64)


#<trian>
print('최적의 매개변수 :',model.best_estimator_) # 가장 좋은 평가 뽑기
# 최적의 매개변수 : SVC(C=1, kernel='linear

print('최적의 파라미터 :',model.best_params_) # 가장 좋은 파람 뽑기
# 최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'}

print('best_score_ :',model.best_score_) # 가장 좋은 점수
# best_score_ : 0.9916666666666668

#<test> 
print('model.score :',model.score(x_test,y_test)) # 테스트한 모델 스코어 (중요)
# model.score : 1.0

y_predict = model.predict(x_test)
print('acc :',accuracy_score(y_test,y_predict))
# acc : 1.0

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :',accuracy_score(y_test,y_pred_best))
# 최적 튠 ACC : 1.0 

# predict / best_estimator_ 값이 같음 -> 최적값 저장됐으니까

print('걸린 시간 :',round(end_time - start_time,2),'초')
# 걸린 시간 : 3.18 초

####################################################################
# print(model.cv_results_)
# 각 파람값 연산된 결과들 출력 52개씩 들어있음

print(pd.DataFrame(model.cv_results_)) # rank_test_score 순위
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=True)) # rank_test_score 순으로 정렬하겠다. / ascending 오름차순 디폴트
print(pd.DataFrame(model.cv_results_).columns) # 컬럼들 출력 / 'split0_test_score', 'split1_test_score', 'split2_test_score','split3_test_score', 'split4_test_score' 각각 교차검증한 결과값

path ='./temp/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=True)\
.to_csv(path +'m14_HalvingGrideSearch1.csv')  # \ 파이썬 문법으로 줄바꿈에 쓴다.

''' 선생님 설명
n_iterations: 3  # 전체 훈련 횟수
n_required_iterations: 4
n_possible_iterations: 3 -> 가능 반복수 
min_resources_: 100    -> 최소 훈련 데이터 갯수
max_resources_: 1437  - > 최대 훈련 데이터 갯수
aggressive_elimination: False
factor: 3   # 엔빵양
----------
iter: 0
n_candidates: 52 # 전체 파라미터 갯수
n_resources: 100 # 0번째 훈련에 쓸 훈련 데이터 갯수
Fitting 5 folds for each of 52 candidates, totalling 260 fits
----------
iter: 1  
n_candidates: 18  # 전체 파라미터 갯수 / factor
n_resources: 300  # min_resources(100) * factor
Fitting 5 folds for each of 18 candidates, totalling 90 fits
----------
iter: 2    
n_candidates: 6  # 18 / factor
n_resources: 900  # 300 * factor
Fitting 5 folds for each of 6 candidates, totalling 30 fits
'''

''' 개인적 정리
n_iterations: 3  # 반복횟수
n_required_iterations: 4
n_possible_iterations: 3 # 3배씩 늘어나는데 4번쨰 훈련까지 하면 리소스 오버됨 그래서 가능 반복횟수는 3
min_resources_: 100
max_resources_: 1437  - > 훈련 데이터 갯수 print(x.shape,x_train.shape) # (1797, 64) (1437, 64)
aggressive_elimination: False
factor: 3   # 디폴트 값 / 파라미터 조절 가능
----------
iter: 0
n_candidates: 52
n_resources: 100 # 최소의 리소스로 훈련 (1437개중 100개)
Fitting 5 folds for each of 52 candidates, totalling 260 fits
----------
iter: 1  # 0번 훈련에서 상위만 추려서 다음 훈련 -> factor: 3이니까 52를 3으로 나눈 18개의 첫 훈련에서의 상위 파라미터
n_candidates: 18
n_resources: 300 # 데이터는 3배로 늘림
Fitting 5 folds for each of 18 candidates, totalling 90 fits
----------
iter: 2    # 마찬가지
n_candidates: 6
n_resources: 900
Fitting 5 folds for each of 6 candidates, totalling 30 fits
'''

'''
GridSearchCV - 260번 훈련 - model.fit에서 나온 내용
걸린 시간 : 9.47 초

HalvingGridSearchCV - 380번 훈련 - model.fit에서 나온 내용
걸린 시간 : 3.95 초
'''


