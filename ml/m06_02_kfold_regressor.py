#for문 돌려
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer,load_wine,load_digits
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler,MinMaxScaler,MaxAbsScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators # 모든 회귀 모델이 들어있음?
import warnings
warnings.filterwarnings('ignore')

datasets = [load_boston(return_X_y=True),
            fetch_california_housing(return_X_y=True),
]
data_name = ['boston','california']

scal = [StandardScaler, MaxAbsScaler, MinMaxScaler , RobustScaler]

scal_name = ['standard','Max','MinMax','Robust']

n_splits = 5 # 디폴트값 5
kfold = KFold(n_splits = n_splits, shuffle=True,random_state=123)

#1.데이터

#2.모델
allAlgorithms = all_estimators(type_filter='regressor') # 모델의 갯수 : 55

max_r2 = 0

for index , value in enumerate(datasets):
    x,y = value
    
    max_score = 0
    max_name = '바보'
    for j, v in enumerate(scal):
        scaler= v        
        for (name,algorithm) in allAlgorithms:
            try:
                model = algorithm()

                scores = cross_val_score(model,x,y,cv=kfold )#n_jobs=-1 병렬처리 안되는 모델도 있어서 빼줘야됨
                results = round(np.mean(scores),4)            
                
            
                if max_score < results :
                    max_score = results
                    max_name = name
                
            except: # 에러뜨면 실행시키는 부분
                continue     
    print('==============',data_name[index],'=================')
    print('최고 모델 :', max_name , max_score ,'스케일러 :',scal_name[j])

'''
데이터 셋 이름
최고 모델 : 이름 값
============== boston =================
최고 모델 : GradientBoostingRegressor 0.8888 스케일러 : Robust
'''
    