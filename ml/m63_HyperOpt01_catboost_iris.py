from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np

from sklearn.datasets import load_diabetes,load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier

x,y = load_iris(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(
    x,y, random_state=337, train_size=0.8
)

scaler= StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from hyperopt import fmin, tpe, Trials , STATUS_OK 
from hyperopt import hp


search_space = { # 파라미터의 범위 / 모든 파라미터가 실수값으로 들어간다.
    'learning_rate' :hp.uniform('learning_rate',0.001,0.2), # quniform은 정수여서 그냥 uniform ( 최소값, 최댓값 )으로
    'depth':hp.quniform('depth',3,16,1),
    'one_hot_max_size': hp.quniform('one_hot_max_size',24,64,1),
    'min_data_in_leaf':hp.quniform('min_data_in_leaf',10,200,1),
    'bagging_temperature':hp.uniform('bagging_temperature',0.5,1),
    'random_strength':hp.uniform('random_strength',0.5,1),
    'l2_leaf_reg':hp.uniform('l2_leaf_reg',0.001,10),

}

def cat_hamsu(search_space):
    params = {
    'iterations' : 10,
    'learning_rate' :search_space['learning_rate'],
    'depth':int(search_space['depth']),
    'l2_leaf_reg':search_space['l2_leaf_reg'], # xgb와 다른부분
    'bagging_temperature':(search_space['bagging_temperature']),
    'random_strength':int(search_space['random_strength']),
    'one_hot_max_size':int(search_space['one_hot_max_size']), #  훈련을 시킬떄의 샘플량 / dropout과 비슷한 개념 / 0~1 사이의 값
    'min_data_in_leaf':int(search_space['min_data_in_leaf']),
    'task_type':'CPU',
    'logging_level':'Silent'
     }
    model = CatBoostClassifier(**params)
    
    
    model.fit(x_train,y_train,
              verbose=0,
              )
    
    y_pred = model.predict(x_test)
    result = accuracy_score(y_test,y_pred)
    
    return result

traial_val = Trials()

#  함수와 파라미터 정의

best = fmin(
    fn = cat_hamsu,
    space= search_space,
    algo = tpe.suggest,
    max_evals=50,
    trials=traial_val,
    rstate =np.random.default_rng(seed=10)
)

print(best)
# {'bagging_temperature': 0.9920056116785902, 'depth': 8.0, 'l2_leaf_reg': 0.16089420995559411, 'learning_rate': 0.1935048484635099, 
# 'min_data_in_leaf': 74.0, 'one_hot_max_size': 37.0, 'random_strength': 0.9972639733314497}

import pandas as pd

results =  [loss_dict['loss'] for loss_dict in traial_val.results]

# for loss_dict in traial_val.results : # 위 코드와 동일함
#     losses.append(loss_dict['loss'])

# 판다스 데이터프레임 형태로 빼 #
# results컬럼에 최소값이 있는 행을 출력
df = pd.DataFrame({'learning_rate':traial_val.vals['learning_rate'],
                   'depth':traial_val.vals['depth'],
                   'l2_leaf_reg':traial_val.vals['l2_leaf_reg'],
                   'bagging_temperature':traial_val.vals['bagging_temperature'],
                   'random_strength':traial_val.vals['random_strength'], 
                   'one_hot_max_size':traial_val.vals['one_hot_max_size'], 
                   'min_data_in_leaf':traial_val.vals['min_data_in_leaf'],                                      
                   'results':results})
print(df)

min_row = df.loc[df['results']==df['results'].min()]
print(min_row)
'''
[50 rows x 8 columns]
    learning_rate  depth  l2_leaf_reg  ...  one_hot_max_size  min_data_in_leaf   results
20       0.193505    8.0     0.160894  ...              37.0              74.0  0.933333
40       0.017173    8.0     7.892777  ...              36.0              53.0  0.933333
41       0.169915    8.0     8.495418  ...              27.0              28.0  0.933333
43       0.168384    7.0     8.297979  ...              27.0              36.0  0.933333
'''

