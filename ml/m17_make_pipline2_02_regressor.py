# 삼중포문 - 데이터 스케일러 모델 파이프라인 모델(3개) 사용
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import time
from sklearn.metrics import r2_score

datasets = [
    load_boston(return_X_y=True),
    fetch_california_housing(return_X_y=True),
]

datasets_name = ['보스턴','캘리포니아']

scalers = [
    MinMaxScaler(),
    MaxAbsScaler(),
    StandardScaler(),
    RobustScaler()
]

scaler_name = ['MINMAX','MAXABS','STANDARD','ROBUST']

models = [
    
    RandomForestRegressor(),
    DecisionTreeRegressor()
]

models_name = ['RandomForestClassifier','DecisionTreeClassifier']

for i , v in enumerate(datasets):
    x,y = v 
    x_train,x_test,y_train,y_test = train_test_split(
        x,y,train_size=0.8, random_state=1234
    )
    
    max_score = 0
    max_name = 'd'
    scal_name = 'f'
    
    # print('==========',datasets_name[i],'==========')
    for j, v1 in enumerate(scalers):
        # scaler = v1 
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.transform(x_test)
        # print('스케일러 :', scaler_name[j])
        
        for k,v2 in enumerate(models):
            model = make_pipeline(v1,v2)
            model.fit(x_train,y_train)
            #4.평가 예측
            result = model.score(x_test,y_test)
            # print('model.score :',result)

            y_pred = model.predict(x_test)
            r2 = r2_score(y_test,y_pred)
            
            if max_score < result:
                max_score = result
                max_name = models_name[k]
                scal_name = scaler_name[j] 
            
        # print('model',models_name[k],'의 accuracy_score :', acc)
    print('====================================')
    print(datasets_name[i],'의 최고 모델 :', max_name , max_score,'최고 스케일러 :' ,scal_name)
                    
'''
====================================
====================================
보스턴 의 최고 모델 : RandomForestClassifier 0.9158885832879415 최고 스케일러 : MINMAX
====================================
캘리포니아 의 최고 모델 : RandomForestClassifier 0.8054921902513543 최고 스케일러 : STANDARD
'''
        
        
        