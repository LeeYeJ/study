# n_component > 0.95이상
# xgboost,gridSearch 또는 RandomSearch 쓸것
#m33_2결과를 뛰어넘을것

    # n_jobs = -1
    # tree_method = 'gpu_hist',
    # predictor = 'gpu_predictor'
    # gpu_id = 0
    
# 과제 시작!

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import time
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


parameters = [
   {'n_estimators':[100,200,300],'learning_rate':[0.1,0.3,0.001,0.01], 'max_depth':[4,5,6]},
   {'n_estimators':[90,100,110],'learning_rate':[0.1,0.001,0.01], 'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1]},
   {'n_estimators':[90,110],'learning_rate':[0.1,0.001,0.4], 'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1],'colsample_bylevel':[0.6,0.7,0.9]},      
]

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  

# Reshape input data
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Normalize input data
x_train = x_train / 255.0
x_test = x_test / 255.0
# x_train,x_test,y_train,y_test = train_test_split(
#     x,y, shuffle=True, random_state=337, test_size=0.2, # stratify=y 사용함으로써 각 라벨값이 골고루 분배된다 
# )   

n_splits = 5 # 디폴트값 5
kfold = KFold(n_splits = n_splits, shuffle=True,random_state=123) 

model = RandomizedSearchCV(XGBClassifier(tree_method = 'gpu_hist',
                    predictor = 'gpu_predictor',
                    gpu_id = 0),
                    parameters, 
                    cv = 5,  # 분류의 디폴트는 StratifiedKFold이다.
                    #  cv = kf,  
                    verbose=1, 
                    refit=True, # 최적의 값을 보관함 / 최적의 값을 출력 -> 통상적으로 True로 함
                    #  refit=False, # 모델이 돌아갈때 최적값을 저장하지 않음 -> False하면 최종 파라미터로 출력
                    n_jobs=-1,
                    )

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

'''




