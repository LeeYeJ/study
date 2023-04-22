from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold,cross_val_predict
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
import numpy as np


x,y = fetch_california_housing(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=True, random_state=12324
)

model = RandomForestRegressor()

kf = KFold(n_splits=5,shuffle=True, random_state=123)

scores = cross_val_score(model,x_train,y_train,cv=5, n_jobs=-1) # cv = 5라고 써도 됨 / 위에서 정의해줘도 되고 /n_jobs=-1 최대 쓰는거임
y_pred = cross_val_predict(model,x_test,y_test,cv =kf)

r2 = r2_score(y_test,y_pred)
print('r2_score :',scores,'\n cross_val_score 평균 :',round(np.mean(scores),4))
print('cross_val_predict_R2 :',r2)

'''
r2_score : [0.80467747 0.79694977 0.82525875 0.79687449 0.79944983] 
 cross_val_score 평균 : 0.8046
cross_val_predict_R2 : 0.7530332691985567
'''

