# 삼중포문 - 데이터 스케일러 모델 파이프라인 모델(3개) 사용
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits,load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline,Pipeline
import time
from sklearn.metrics import accuracy_score

datasets = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_wine(return_X_y=True),
    load_digits(return_X_y=True)
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

models = [
    SVC(),
    RandomForestClassifier(),
    DecisionTreeClassifier()
]

models_name = ['SVC','RandomForestClassifier','DecisionTreeClassifier']

for i , v in enumerate(datasets):
    x,y = v 
    x_train,x_test,y_train,y_test = train_test_split(
        x,y,train_size=0.8, random_state=124,
    )
    max_score = 0  # 데이터 쪽에서 종속시켜야됨
    max_name = 'd' # 데이터 쪽에서 종속시켜야됨
    scal_name = 'f' # 데이터 쪽에서 종속시켜야됨  그래야 아래서부터 진행됨
    
    # print('==========',datasets_name[i],'==========')
    for j, v1 in enumerate(scalers):
        # scaler = v1 
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.transform(x_test)
        # # print('스케일러 :', scaler_name[j])
        for k,v2 in enumerate(models):
            model = Pipeline([('v1',v1),('v2',v2)])
            model.fit(x_train,y_train)
            #4.평가 예측
            result = model.score(x_test,y_test)
            # print('model.score :',result)

            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test,y_pred)
            # print('accuracy_score :', acc)
            
            if max_score < result:
                max_score = result
                max_name = models_name[k]
                scal_name = scaler_name[j] 
            
            # print('model',models_name[k],'의 accuracy_score :', acc,'scla :',scal_name)
    print('====================================')
    print(datasets_name[i],'의 최고 모델 :', max_name , max_score,'최고 스케일러 :' ,scal_name)
                    
'''
====================================
아이리스 의 최고 모델 : SVC 0.9333333333333333 최고 스케일러 : ROBUST
====================================
암 의 최고 모델 : SVC 0.9824561403508771 최고 스케일러 : MAXABS
====================================
와인 의 최고 모델 : SVC 1.0 최고 스케일러 : MINMAX
====================================
디짓 의 최고 모델 : SVC 0.9833333333333333 최고 스케일러 : MINMAX
'''
        
        
        