#pca - 차원(컬럼) 축소(압축)하는 개념
# 쓸모없는 컬럼들은 압축되면 성능이 좋아질수도있다
# 차원을 축소해서 생긴 y값 -> 타겟값이 생기니까 비지도이다. / -> 스케일러의 개념도?

# 컬럼간의 좌표를 찍었을때 그려지는 직선 위로 데이터들의 좌표가 매핑된다.

# 차원이 축소될때마다 그 직선위에 수직으로 그려진다.

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits,load_breast_cancer,fetch_california_housing
# from sklearn.datasets import load_breast_cancer,load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1.데이터

datasets = fetch_california_housing()
print(datasets.feature_names) # 판다스에서는 컬럼 이름 -> columns로 확인 / sklearn은 feature_names로 확인 (실무에선 사이킷런으로 확인할 일 없을듯?)
x =datasets['data'] # 넘파이
y = datasets.target
print(x.shape) #(20640, 8)

max_score = 0
max_num = 'f'
for n in range(8, 0, -1):
    pca = PCA(n_components=n)
    x = pca.fit_transform(x)
    # print(f"n_components={n}: x_transformed shape={x_transformed.shape}")
    
    x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8,random_state=123,shuffle=True
    )

    #2. 모델
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=123)

    #3.훈련
    model.fit(x_train,y_train)

    #4.평가 예측
    results = model.score(x_test,y_test)
    print('결과  :', results)
    print(f"n_components={n}: 결과={results}")
    
    
    if max_score < results:
        max_score = results
        max_num =  n_components={n}  
print('최고의 components 수는',max_num,'이고 결과는',max_score,'이다')
'''
결과  : 0.7825857242412009
n_components=8: 결과=0.7825857242412009
결과  : 0.7786727671384369
n_components=7: 결과=0.7786727671384369
결과  : 0.7018597110810503
n_components=6: 결과=0.7018597110810503
결과  : 0.5918722922244304
n_components=5: 결과=0.5918722922244304
결과  : 0.3241551445937575
n_components=4: 결과=0.3241551445937575
결과  : 0.0789494633195541
n_components=3: 결과=0.0789494633195541
결과  : 0.046317003872303086
n_components=2: 결과=0.046317003872303086
결과  : -0.4412309439201201
n_components=1: 결과=-0.4412309439201201
최고의 components 수는 {8} 이고 결과는 0.7825857242412009 이다
'''
