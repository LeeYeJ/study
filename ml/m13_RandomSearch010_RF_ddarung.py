from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, fetch_covtype,load_breast_cancer,load_boston
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time
import pandas as pd
from sklearn.metrics import accuracy_score


# 모델 RandomForestClassifier

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
    {'min_samples_split':[2,3,5,10]}
]


path='./_data/ddarung/'
path_save='./_save/ddarung/'

train_csv=pd.read_csv(path+'train.csv', index_col=0)

print(train_csv.shape) #(1459, 10)

test_csv=pd.read_csv(path +'test.csv', index_col=0)

print(test_csv.shape) #(715, 9)

#결측치 제거
print(train_csv.isnull().sum())
train_csv=train_csv.dropna()
print(train_csv.isnull().sum())

#데이터분리!!!!!!!!!!!!!!! 외워 좀!! 
x=train_csv.drop(['count'],axis=1)
y=train_csv['count']

print(train_csv.shape)
print(test_csv.shape)

#데이터분리

x_train,x_test,y_train,y_test=train_test_split(
 x,y,shuffle=True,random_state=4897567,test_size=0.1
)


scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
scaler.fit(x_train) # fit의 범위가 x_train이다
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

#test 파일도 스케일링 해줘야됨!!!!!!!!!
test_csv=scaler.transform(test_csv)

kf = KFold(n_splits=5,shuffle=True, random_state=3324)

# 데이터 split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True
)
model = RandomizedSearchCV(RandomForestClassifier(),
                     parameters, 
                     cv = 5,  # 분류의 디폴트는 StratifiedKFold이다.
                    #  cv = kf,  
                     verbose=1, 
                     refit=True, # 최적의 값을 보관함 / 최적의 값을 출력 -> 통상적으로 True로 함
                    #  refit=False, # 모델이 돌아갈때 최적값을 저장하지 않음 -> False하면 최종 파라미터로 출력
                     n_jobs=-1)

#3.컴파일 훈련
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()

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
print('acc :',r2_score(y_test,y_predict))
# acc : 1.0

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :',r2_score(y_test,y_pred_best))
# 최적 튠 ACC : 1.0 

# predict / best_estimator_ 값이 같음 -> 최적값 저장됐으니까

print('걸린 시간 :',round(end_time - start_time,2),'초')
# 걸린 시간 : 3.18 초
'''
최적의 매개변수 : RandomForestClassifier(min_samples_leaf=10, min_samples_split=10)
최적의 파라미터 : {'min_samples_split': 10, 'min_samples_leaf': 10}
best_score_ : 0.04016582914572865
model.score : 0.012048192771084338
acc : 0.5791214129050415
최적 튠 ACC : 0.5791214129050415
걸린 시간 : 9.93 초
'''