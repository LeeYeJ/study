# 모델에서 스케일러를 동시에~
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline,Pipeline 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#1.데이터
x,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8, shuffle=True, random_state=337
)

parameters = [
    {'rf__n_estimators':[100,200], 'rf__max_depth':[6,10,12],'rf__min_samples_leaf':[3,10]},
    {'rf__max_depth':[6,8,10,12],'rf__min_samples_leaf':[3,5,7,10]},
    {'rf__min_samples_leaf':[3,5,7,10],'rf__min_samples_split':[2,3,5,10]},
    {'rf__min_samples_split':[2,3,5,10],'rf__max_depth':[6,8,10,12]}
]

#2.모델

pipe = Pipeline([('std',StandardScaler()),('rf',RandomForestClassifier())])  # 리스트 안에 튜플 형태로 만들고 각 이름 명시까지 해줘야됨

model = GridSearchCV(pipe, parameters, cv=5,verbose=1, n_jobs=-1) # 파이프의 모델을 가져온다 / 랜포 모델의 파람을 파이프라인 파라미터의 형태로 바꿔준다.(튜플에 정의해둔 이름을 앞에 사용)

#3.훈련
model.fit(x_train,y_train)

#4.평가 예측
result = model.score(x_test,y_test)
print('model.score :',result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print('accuracy_score :', acc)


