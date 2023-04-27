
# SelectFromModel 사용해서 피처 삭제하면서 결과 뽑는 for문 아래 만들어봤음
# 그렇다면 SelectFromModel 모델에선 다른 모델을 사용해도 가능할까? 응 가능해
# ValueError: Either fit the model before transform or set "prefit=True" while passing the fitted estimator to the constructor. -> 버전 문제 해결 업글
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,RobustScaler # RobustScaler는 중앙값과 IQR을 사용하여 스케일링해서 이상치에 덜 민감하다.
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import accuracy_score ,r2_score, mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# featureN = datasets['feature_names']
# print(featureN)
# '''
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# '''

 
x_train,x_test,y_train,y_test = train_test_split(
    x,y, random_state=337, train_size=0.8
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
        'n_estimators': 1000,  # 디폴트 100 / 1~ inf / 정수
        'learning_rate':  0.1, # 디폴트 0.3 / 0~1 / eta
        'max_depth': 2, # 디폴트 6 / 
        'gamma': 1,
        'min_child_weight': 1,
        'subsample': 0.7, 
        'colsample_bytree': 0.2,
        'colsample_bylevel': 1., # . float형으로 변환?
        'colsample_bynode': 1.,
        'reg_alpha' : 0.1,
        'reg_lambda': 0.1,
        'random_state':337,
        # 'eval_metric' : 'error' 
}


#2. 모델
# model = XGBClassifier(**parameters)
model = XGBRegressor()


# 3. 훈련
model.set_params(early_stopping_rounds =10,**parameters,eval_metric = 'rmse' ,
                #  eval_metric='rmse'
                 ) # 파람도 여기서 먹힘

model.fit(x_train,y_train,
            eval_set = [(x_train,y_train),(x_test,y_test)], # <- early_stopping을 쓰기 위한 validation 데이터가 됨 / 각각 훈련 로스와 발리 로스 출력 가능
            verbose=1,
            # eval_metric = 'logloss', # 이진분류
            # eval_metric = 'error', # 이진분류에서 쓰는것
            # eval_metric = 'auc'    # 이진분류   
            # eval_metric = 'merror' # 다중분류에서 쓰는것 mlogloss ( m -> multi )
            # eval_metric = 'rmse' #'mae','rmsle'....  #분류 데이터도 회귀로 가능하긴함 반대로 회귀는 분류 X
            )

# Must have at least 1 validation dataset for early stopping. ->validation데이터 필요

# 4. 평가 예측 (grid)
# print('최상의 매개변수 :',model.best_params_)
# print('최상의 점수 :',model.best_score_)

results = model.score(x_test,y_test)
print('최종 점수 :', results) # r2스코어

y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print('r2 : ',r2)

mse = mean_squared_error(y_test,y_pred)
print('RMSE :',np.sqrt(mse))

##########################################
print(model.feature_importances_)
'''
[0.04063273 0.03594474 0.22723462 0.14466234 0.03031579 0.04569667
 0.06766509 0.16443627 0.1592291  0.08418266]
'''
thresholds = model.feature_importances_
thresholds = np.sort(model.feature_importances_) # 오름 차순
print(thresholds)
'''
[0.03031579 0.03594474 0.04063273 0.04569667 0.06766509 0.08418266
 0.14466234 0.1592291  0.16443627 0.22723462]
'''
from sklearn.feature_selection import SelectFromModel

for i in  thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False) # prefit 사전훈련된 가중치를 사용하겠냐 / False면 다시 훈련 /// threshold 특정값 이상의 값을 뽑아주겠다 (기준을 잡음)    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print('변형된 x_train',select_x_train.shape,'변형된 x_test',select_x_test.shape)
    
    selection_model = LinearRegression()
    # selection_model.set_params(early_stopping_rounds = 10, **parameters,eval_metric ='rmse')
    selection_model.fit(select_x_train,y_train,
                        # eval_set = [(select_x_train,y_train),(select_x_test,y_test)],
                        
                        )
    
    select_y_pred = selection_model.predict(select_x_test)
    score = r2_score(y_test,select_y_pred) 
    
    print('Tresh=%.3f, n=%d ,R2:%.2f%%'%(i,select_x_train.shape[1], score*100)) # 파이썬 옛날 문법 / 앞의 세 %붙은 놈들이 뒤의 각 자리와 대응됨

    
        

