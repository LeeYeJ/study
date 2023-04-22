
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, fetch_covtype,load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import time
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Conv2D,Flatten,LSTM,Conv1D,Flatten
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
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

path='./_data/dacon_diabets/'
path_save='./_save/dacon_diabets/'

train_csv=pd.read_csv(path+'train.csv',index_col=0) #@@인덱스!

test_csv=pd.read_csv(path+'test.csv',index_col=0)


x=train_csv.drop(['Outcome'],axis=1) #@@@@
y=train_csv['Outcome']


x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=4545451,train_size=0.9,
)

print(x_train.shape) #(586, 8)
print(x_test.shape) #(66, 8)
print(test_csv.shape) #(116, 8)


scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

test_csv=scaler.transform(test_csv)

n_splits = 5 # 디폴트값 5
kfold = KFold(n_splits = n_splits, shuffle=True,random_state=123) 

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
print('acc :',accuracy_score(y_test,y_predict))
# acc : 1.0

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :',accuracy_score(y_test,y_pred_best))
# 최적 튠 ACC : 1.0 

# predict / best_estimator_ 값이 같음 -> 최적값 저장됐으니까

print('걸린 시간 :',round(end_time - start_time,2),'초')
# 걸린 시간 : 3.18 초
'''
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수 : RandomForestClassifier(max_depth=8, min_samples_leaf=5)
최적의 파라미터 : {'min_samples_leaf': 5, 'max_depth': 8}
best_score_ : 0.7593944661741272
model.score : 0.803030303030303
acc : 0.803030303030303
최적 튠 ACC : 0.803030303030303
걸린 시간 : 6.7 초
'''
