# 최소값을 찾는거다!
# 베이지안 옵티마이제이션은 최대값을 찾는거다.

import hyperopt
print(hyperopt.__version__) #0.2.7

from hyperopt import hp
import numpy as np

search_space = { # 파라미터 모아둔것
    'x1' : hp.quniform('x1',-10,10,1), # -10 ~ 10까지 1단위로 찾는다. / uniform
    'x2' : hp.quniform('x2',-15,15,1)
    # hp.quniform(label,low,high, q(간격)) 정규분포 형태로 서치한다.
}

print(search_space)

def objective_func(search_space):  # 목적 함수 만들어둔것
    x1 = search_space['x1']
    x2 = search_space['x2']
    
    return_value = x1**2 - 20*x2
    
    return return_value
    # 권장 리턴 방식 return {'loss': return_value, 'status': STATUS_OK}

from hyperopt import fmin, tpe, Trials , STATUS_OK # 최소값 찾는 fmin 

traial_val=Trials()

best = fmin(
    fn= objective_func,
    space = search_space,# space => 파라미터라는 의미
    algo = tpe.suggest, # 알고리즘 디폴트
    max_evals=100,  # 베이지안 옵티마이제이션의 n_iter와 같은 것임./ 몇번 반복해주겠냐
    trials= traial_val, # Trials 결과값 저장 (히스토리와 같은것)
    rstate= np.random.default_rng(seed=10) # 랜덤스테이트와 같은 것
)

print('best',best)

print(traial_val.results)
'''
[{'loss': -216.0, 'status': 'ok'}..., {'loss': -104.0, 'status': 'ok'}]
'''

print(traial_val.vals)
'''
{'x': [-2.0, -5.0, 7.0, 10.0, 10.0,...],
'x2': [11.0, 10.0, -4.0, -5.0,]}
'''
## pandas 데이터프레임에 trial_val.vals를 넣어라 ##
# traial_val_v = pd.DataFrame(traial_val.vals)
# print(traial_val_v)

import pandas as pd

results =  [loss_dict['loss'] for loss_dict in traial_val.results]

# for loss_dict in traial_val.results : # 위 코드와 동일함
#     losses.append(loss_dict['loss'])
    
df = pd.DataFrame({'x1':traial_val.vals['x1'],
                   'x2':traial_val.vals['x2'],
                   'results':results})
print(df)

'''

      x1    x2  results
0   -2.0  11.0   -216.0
1   -5.0  10.0   -175.0
2    7.0  -4.0    129.0
3   10.0  -5.0    200.0
4   10.0  -7.0    240.0
..   ...   ...      ...
95  -1.0  11.0   -219.0
96  -9.0  -1.0    101.0
97   2.0   1.0    -16.0
98  -4.0   8.0   -144.0
99   4.0   6.0   -104.0

'''


