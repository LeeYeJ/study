#m17 카피
#model.feature_importances_ 컬럼의 중요도 -> acc를 신뢰한다는 가정하에 보는거임
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,r2_score
from sklearn.pipeline import make_pipeline  
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score

datasets = [
    load_boston,
    fetch_california_housing,
]

datasets_name = [
    'boston',
    'california'
]

trees = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier()
]

trees_name = [
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'XGBClassifier'
]

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

for i, v in datasets:
    
    x,y = v(return_X_y=True)

    x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8, shuffle=True, random_state=337)

        
    for i,v in enumerate(trees):
        model = v
    #2.모델 # 이 네가지 모델은 모두 트리계열이다.
    # model = DecisionTreeClassifier()
    # model = RandomForestClassifier()
    # model = GradientBoostingClassifier()
    # model = XGBClassifier()
    #3.훈련
        model.fit(x_train,y_train)

    #4.평가 예측
    # result = model.score(x_test,y_test)
    # print('model.score :',result)

        y_pred = model.predict(x_test)
        acc = r2_score(y_test,y_pred)
        print('accuracy_score :', acc)
    print(model,':',model.feature_importances_) 

    #model.feature_importances_ 트리 계열에만 있음 /  낮은 것부터 제거
    # if i != 3:
    #     print(model,':',model.feature_importances_) 
    #     print('=============================')
    # else:
    #     print('xgboost() :',model.feature_importances_)
        

