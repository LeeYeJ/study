from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import warnings
warnings.filterwarnings('ignore')

x,y = load_diabetes(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(
    x,y, random_state=337, train_size=0.8
)

scaler= StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from hyperopt import fmin, tpe, Trials , STATUS_OK 
from hyperopt import hp


search_space = { # 파라미터의 범위 / 모든 파라미터가 실수값으로 들어간다.
    'learning_rate' :hp.uniform('learning_rate',0.001,1), # quniform은 정수여서 그냥 uniform ( 최소값, 최댓값 )으로
    'max_depth':hp.quniform('max_depth',3,16,1),
    'num_leaves': hp.quniform('num_leaves',24,64,1),
    # 'min_child_samples':hp.quniform(10,200,1),
    # 'min_child_weight':hp.quniform(1,50,1),
    'subsample':hp.uniform('subsample',0.5,1),
    # 'colsample_bytree':hp.uniform(0.5,1),
    # 'max_bin':hp.quniform(10,500,1),
    # 'reg_lambda':hp.uniform(0.001,10),
    # 'reg_alpha':hp.uniform(0.01,50)
}
# hp.quniform(label,low,high,q) : 최소부터 최대까지 q간격 / q 소수형태임 (1이 1.0임 따라서 int처리 해줘야됨)
# hp.uniform(label,low,high) :최소부터 최대까지 정규분포 간격 (정규분포로 중심부가 많이 제공됨)
# hp.randint(label, upper): 0부터 최대값 upper까지 random한 정수값
# hp.loguniform(label,low,high) :exp(uniform(low,high))값 반환 ,이거 역시 정규분포 형태 <- 다시 지수로 변환

def lgb_hamsu(search_space):
    params = {
    'n_estimators' : 1000,
    'learning_rate' :search_space['learning_rate'],
    'max_depth':int(search_space['max_depth']),
    'num_leaves':int(search_space['num_leaves']), # xgb와 다른부분
    # 'min_child_samples':int(round(min_child_samples)),
    # 'min_child_weight':int(round(min_child_weight)),
    'subsample':search_space['subsample'], #  훈련을 시킬떄의 샘플량 / dropout과 비슷한 개념 / 0~1 사이의 값
    # 'colsample_bytree':colsample_bytree,
    # 'max_bin':max(int(round(max_bin)),10),  # 최대가 10 이상이다 / max로 10 이상 뽑을수있음
    # 'reg_lambda':max(reg_lambda,0), # 무조건 양수만 빼야됨
    # 'reg_alpha':reg_alpha,
     }
    model = LGBMRegressor(**params)
    
    
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_pred = model.predict(x_test)
    result = mean_squared_error(y_test,y_pred)
    
    return result

traial_val = Trials()

#  함수와 파라미터 정의

best = fmin(
    fn = lgb_hamsu,
    space= search_space,
    algo = tpe.suggest,
    max_evals=50,
    trials=traial_val,
    rstate =np.random.default_rng(seed=10)
)

print(best)
# {'learning_rate': 0.2509097581813602, 'max_depth': 7.0, 'num_leaves': 41.0, 'subsample': 0.8759587138041729}

import pandas as pd

results =  [loss_dict['loss'] for loss_dict in traial_val.results]

# for loss_dict in traial_val.results : # 위 코드와 동일함
#     losses.append(loss_dict['loss'])

# 판다스 데이터프레임 형태로 빼 #
# results컬럼에 최소값이 있는 행을 출력
df = pd.DataFrame({'learning_rate':traial_val.vals['learning_rate'],
                   'max_depth':traial_val.vals['max_depth'],
                   'num_leaves':traial_val.vals['num_leaves'],
                   'subsample':traial_val.vals['subsample'],                   
                   'results':results})
print(df)

min_row = df.loc[df['results']==df['results'].min()]
print(min_row)
'''
    learning_rate  max_depth  num_leaves  subsample      results
33        0.25091        7.0        41.0   0.875959  2754.155255
'''

