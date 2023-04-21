# 예외처리
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators # 모든 회귀 모델이 들어있음?
import warnings
warnings.filterwarnings('ignore')
import sklearn as sk
print(sk.__version__)

#1.데이터
x,y = load_digits(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y, random_state=123, test_size=0.2
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2.모델
# model = RandomForestRegressor(n_jobs=4) #n_estimators-> 에포 / n_jobs =4 코어 4개 다씀 속도 빨라짐
allAlgorithms = all_estimators(type_filter='regressor',) 
# allAlgorithms = all_estimators(type_filter='classifier',) # 모델의 갯수 : 41

print('allAlgorithms:',allAlgorithms) # 리스트 안에 두개씩 튜플 형태로 들어있다.
print('모델의 갯수 :',len(allAlgorithms)) #모델의 갯수 : 55

max_r2 = 0
for (name,algorithm) in allAlgorithms:
    try:
        model = algorithm()
        # 3. 훈련 (에포가 디폴트 100 -> 모델마다 에포 이름 다름)
        model.fit(x_train,y_train)

        # 4. 평가 예측
        results = model.score(x_test,y_test)
        print(name,'의 정답률 :', results)

        if max_r2 < results :
            max_r2 = results
            max_name = name
                
        
        # y_predict = model.predict(x_test)
        # print(y_test.dtype) # float64
        # print(y_predict.dtype) # float64
        # aaa = r2_score(y_test,y_predict)
        # print('r2 :', aaa) 
        
    except: # 에러뜨면 실행시키는 부분
        print(name,'은(는) 에러')      
print('====================================')
print('최고 모델 :', max_name , max_r2 )

#results는 모델에서 알아서 r2로 뽑아준다.
'''
====================================
최고 모델 : ExtraTreesRegressor 0.8942189262918134

'''


