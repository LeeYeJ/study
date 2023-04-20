# 파라미터의 경우의 수만큼 다 돌음

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# cross_validation + 하이퍼파라미터
from sklearn.model_selection import GridSearchCV
import time

# 1. 데이터
x,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, # stratify=y 사용함으로써 각 라벨값이 골고루 분배된다 
)    

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True,random_state=337)

parameters = [
    {'C':[1,10,100,1000],'kernel':['linear'],'degree':[3,4,5]}, # 12번 돌음
    {'C':[1,10,100],'kernel':['rbf','linear'],'gamma':[0.001,0.0001]}, # 12번 돌음
    {'C':[1,10,100,1000],'kernel':['sigmoid'],
     'gamma':[0.01,0.001,0.0001],'degree':[3,4]}, # 24번 돌음
    {'C':[0.1,1],'gamma':[1,10]} # 4번 돌음
]          # 총 52번 돌음

#2.모델
# GridSearchCV의 CV는 디폴트가 StratifiedKFold이다. 그래서 아래 파람 조절을 cv = 5를 기본으로 줬을떄 더 잘 나올수있다.
model = GridSearchCV(SVC(),
                     parameters, 
                     cv = 5,  # 분류의 디폴트는 StratifiedKFold이다.
                    #  cv = kf,  
                     verbose=1, 
                    #  refit=True, # 최적의 값을 보관함 / 최적의 값을 출력 -> 통상적으로 True로 함
                    #  refit=False, # 모델이 돌아갈때 최적값을 저장하지 않음 -> False하면 최종 파라미터로 출력
                     n_jobs=-1) # GridSearchCV 안에 사용할 모델과 파라미터 정의


#3.컴파일 훈련
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()

#최적의 파라미터 찾기

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