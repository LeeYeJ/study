#m17 카피
#model.feature_importances_ 컬럼의 중요도 -> acc를 신뢰한다는 가정하에 보는거임
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline  
from sklearn.svm import SVC

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

#1.데이터
x,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
x,y,train_size=0.8, shuffle=True, random_state=337)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
    acc = accuracy_score(y_test,y_pred)
    print('accuracy_score :', acc)

    #model.feature_importances_ 트리 계열에만 있음 /  낮은 것부터 제거
    if i !=3:
        print(model,':',model.feature_importances_) 
        print('=============================')
    else:
        print('xgboost() :',model.feature_importances_)
        
'''
accuracy_score : 0.9333333333333333
DecisionTreeClassifier() : [0.01671193 0.03342386 0.9139125  0.03595171]
=============================
accuracy_score : 0.9666666666666667
RandomForestClassifier() : [0.04637402 0.02253518 0.44805613 0.48303467]
=============================
accuracy_score : 0.9666666666666667
GradientBoostingClassifier() : [0.0041898  0.01484657 0.68897641 0.29198722]
=============================
accuracy_score : 0.9666666666666667
xgboost() : [0.01794496 0.01218657 0.8486943  0.12117416]
'''
    
   
        
#DecisionTreeClassifier() : [0.         0.05013578 0.9139125  0.03595171] -> 컬럼의 중요도 -> acc를 신뢰한다는 가정하에 보는거임
#RandomForestClassifier() : [0.12696916 0.03399441 0.41538916 0.42364727]
#GradientBoostingClassifier() : [0.0044375  0.01496767 0.62964428 0.35095055]
#XGBClassifier() : [0.01794496 0.01218657 0.8486943  0.12117416]
