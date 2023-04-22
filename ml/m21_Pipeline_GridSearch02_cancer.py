
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer,load_wine,load_digits,fetch_covtype
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline,Pipeline 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

#1.데이터
x,y = load_breast_cancer(return_X_y=True)

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

#3.컴파일 훈련
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()

print('걸린 시간 :',round(end_time - start_time,2),'초')

print('최적의 매개변수 :',model.best_estimator_) # 가장 좋은 평가 뽑기

print('최적의 파라미터 :',model.best_params_) # 가장 좋은 파람 뽑기

print('best_score_ :',model.best_score_) # 가장 좋은 점수

print('model.score :',model.score(x_test,y_test)) # 테스트한 모델 스코어 (중요)

y_predict = model.predict(x_test)
print('acc :',accuracy_score(y_test,y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :',accuracy_score(y_test,y_pred_best))


print('걸린 시간 :',round(end_time - start_time,2),'초')
'''
Fitting 5 folds for each of 60 candidates, totalling 300 fits
걸린 시간 : 15.89 초
최적의 매개변수 : Pipeline(steps=[('std', StandardScaler()),
                ('rf', RandomForestClassifier(max_depth=6))])
최적의 파라미터 : {'rf__max_depth': 6, 'rf__min_samples_split': 2}
best_score_ : 0.956043956043956
model.score : 0.956140350877193
acc : 0.956140350877193
최적 튠 ACC : 0.956140350877193
걸린 시간 : 15.89 초
'''