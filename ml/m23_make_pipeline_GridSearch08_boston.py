# 랜덤 서치, 그리드 서치 , 할빙 그리드 서치를 
# for문으로 한방에 넣어라
# fetchcovtype 랜덤 할빙 n_iter =5 , cv =2

import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,r2_score
from sklearn.pipeline import make_pipeline,Pipeline 
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
import time

parameters = [ # 모델의 이름을 모두 소문자로 사용한다.
    {'randomforestclassifier__n_estimators':[100,200], 'randomforestclassifier__max_depth':[6,10,12],'randomforestclassifier__min_samples_leaf':[3,10]},
    {'randomforestclassifier__max_depth':[6,8,10,12],'randomforestclassifier__min_samples_leaf':[3,5,7,10]},
    {'randomforestclassifier__min_samples_leaf':[3,5,7,10],'randomforestclassifier__min_samples_split':[2,3,5,10]},
    {'randomforestclassifier__min_samples_split':[2,3,5,10],'randomforestclassifier__max_depth':[6,8,10,12]}
]

datasets = [
    load_boston(return_X_y=True),
    fetch_california_housing(return_X_y=True),
]

datasets_name = ['아이리스','암','와인','디짓']

scalers = [
    MinMaxScaler(),
    MaxAbsScaler(),
    StandardScaler(),
    RobustScaler()
]

scaler_name = [
    'MINMAX'
    ,'MAXABS'
    ,'STANDARD'
    ,'ROBUST'
]

gs = [
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV
]

gs_name = [
    'GridSearchCV',
    'RandomizedSearchCV',
    'HalvingGridSearchCV'
]

for i, v in enumerate(datasets):
    x,y = v
    x_train,x_test,y_train,y_test = train_test_split(
        x,y, train_size=0.8, random_state=1234
    )
        
    max_score=0
    max_name = 'd'
    scal_name='0'  
    print('===============',datasets_name[i],'==================')
    for j,v1 in enumerate(scalers):
        for k, v2 in enumerate(gs):
            pipe = make_pipeline(v1,RandomForestClassifier())
            model = v2(pipe, parameters, cv=5,verbose=1, n_jobs=-1)
            #3.훈련
            model.fit(x_train,y_train)
            #4.평가 예측
            result = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            acc = r2_score(y_test,y_pred)
            # print('accuracy_score :', acc)
            
            if max_score < result:
                max_score = result
                max_name = gs_name[k]
                scal_name = scaler_name[j]
    print(max_name,'와', scal_name,'의 최고 model.score은',max_score,'이다')
'''
                            .
                            .
                            .
======================= 아이리스 ===========================
GridSearchCV 와 MINMAX 의 최고 model.score은 1.0 이다
                            .
                            .
                            .
                            
                            .
                            .
                            .
                            
                            .
                            .
                            .
'''
            