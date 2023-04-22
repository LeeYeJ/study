from sklearn.model_selection import train_test_split, cross_val_score, KFold
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

model = RandomForestClassifier()

scores = cross_val_score(model,x,y,cv=5, n_jobs=-1) # cv = 5라고 써도 됨 / 위에서 정의해줘도 되고 /n_jobs=-1 최대 쓰는거임
print('ACC :',scores,'\n cross_val_score 평균 :',round(np.mean(scores),4))