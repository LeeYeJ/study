#pca - 차원(컬럼) 축소(압축)하는 개념
# 쓸모없는 컬럼들은 압축되면 성능이 좋아질수도있다
# 차원을 축소해서 생긴 y값 -> 타겟값이 생기니까 비지도이다. / -> 스케일러의 개념도?

# 컬럼간의 좌표를 찍었을때 그려지는 직선 위로 데이터들의 좌표가 매핑된다.

# 차원이 축소될때마다 그 직선위에 수직으로 그려진다.

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits,load_breast_cancer
# from sklearn.datasets import load_breast_cancer,load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1.데이터

path = './_data/kaggle_bike/'
path_save='./_save/kaggle_bike/'
train_csv= pd.read_csv(path +'train.csv', index_col=0)
test_csv=pd.read_csv(path + 'test.csv', index_col=0)

#데이터 결측치 제거
print(train_csv.isnull().sum()) # 결측값 없음

#데이터 x,y분리
x=train_csv.drop(['count','casual','registered'], axis=1)
print(x.columns)
y=train_csv['count']
print(y)

print(x.shape) # (10886, 8)

# 데이터 split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True
)

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
결과  : 0.2895771892656308
n_components=8: 결과=0.2895771892656308
결과  : 0.2938420048759155
n_components=7: 결과=0.2938420048759155
결과  : 0.2664999926233613
n_components=6: 결과=0.2664999926233613
결과  : 0.23961437109608652
n_components=5: 결과=0.23961437109608652
결과  : 0.23390529920394865
n_components=4: 결과=0.23390529920394865
결과  : 0.12472498676921084
n_components=3: 결과=0.12472498676921084
결과  : 0.11616342655473855
n_components=2: 결과=0.11616342655473855
결과  : -0.15106887979914108
n_components=1: 결과=-0.15106887979914108
최고의 components 수는 {7} 이고 결과는 0.2938420048759155 이다
'''
