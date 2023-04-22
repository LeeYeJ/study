import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer,load_wine,load_digits
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler,MinMaxScaler,MaxAbsScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators # 모든 회귀 모델이 들어있음?
import warnings
warnings.filterwarnings('ignore')

datasets = [load_iris(return_X_y=True),
            load_breast_cancer(return_X_y=True),
            load_wine(return_X_y=True),
            load_digits(return_X_y=True)
]
data_name = ['iris','cancer','wine','digits']

n_splits = 5 # 디폴트값 5
kfold = KFold(n_splits = n_splits, shuffle=True,random_state=123)

#1.데이터

#2.모델



for index , value in enumerate(datasets):
    
    x,y = value
    x_train, x_test, y_train, y_test = train_test_split(
        x,y, shuffle=True
    )
    
    allAlgorithms = all_estimators(type_filter='classifier') # 모델의 갯수 : 55
    max_score = 0
    max_name = '바보'
    for (name,algorithm) in allAlgorithms:
        try:
            model = algorithm()

            scores = cross_val_score(model,x_train,y_train,cv=kfold )#n_jobs=-1 병렬처리 안되는 모델도 있어서 빼줘야됨
            
            results = round(np.mean(scores),4)
            y_predict = cross_val_predict(model,x_test,y_test,cv=kfold)
            acc = accuracy_score(y_test,y_predict) 
            print(name,'의 정답률 :', results,'acc :',acc) 
            
            if max_score < results :
               max_score = results
               max_name = name                
               
        except: # 에러뜨면 실행시키는 부분
            continue     
    print('==============',data_name[index],'=================')
    print('최고 모델 :', max_name , max_score ,'acc :', acc)


'''

'''
    