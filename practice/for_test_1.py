import numpy as np
from sklearn.datasets import load_breast_cancer,load_iris,load_wine
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler, RobustScaler


# 데이터
data_list = [load_breast_cancer(return_X_y=True),
             load_iris(return_X_y=True),
             load_wine(return_X_y=True)]

data_name = ['암','아이리스','와인']

model_list = [RandomForestClassifier(),DecisionTreeClassifier()]

model_name = ['random','tree']

scaler = [StandardScaler(),MaxAbsScaler(),MinMaxScaler(),RobustScaler()]

scaler_name=['standard','Max','MinMax','Robuster']

for i , value in enumerate(data_list):
    x,y = value
    x_train,x_test,y_train,y_test = train_test_split(
        x,y, shuffle=True,random_state=1234
    )
    print('===================')
    print(data_name[i])
    
    scal_n = ''
    for k, val in enumerate(scaler):
        scaler = val
        scaler.fit_transform(x_train)
        scaler.transform(x_test)
        
        max_score =0
        max_name=''
        for j, v  in enumerate(model_list):
            model = v
            model.fit(x,y)
            result = model.score(x,y)
            print(model_name[j],'model_score :',result)
            y_pred = np.round(model.predict(x))
            acc = accuracy_score(y,y_pred)
            print(model_name[j],'acc:',acc)
            
            if max_score < result:
               max_score = result
               max_name = j
               
               
               
    print('최고 모델 :',max_name , max_score )
    print('최고 스케일러 :' , scal_n)               
