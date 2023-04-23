#pca - 차원(컬럼) 축소(압축)하는 개념
# 쓸모없는 컬럼들은 압축되면 성능이 좋아질수도있다
# 차원을 축소해서 생긴 y값 -> 타겟값이 생기니까 비지도이다. / -> 스케일러의 개념도?
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1.데이터

datasets = load_breast_cancer()
print(datasets.feature_names) # 판다스에서는 컬럼 이름 -> columns로 확인 / sklearn은 feature_names로 확인 (실무에선 사이킷런으로 확인할 일 없을듯?)
x =datasets['data'] # 넘파이
y = datasets.target
print(x.shape)  # (569, 30)

max_score = 0
max_num = 'f'
for n in range(30, 0, -3):
    pca = PCA(n_components=n)
    x = pca.fit_transform(x)
    # print(f"n_components={n}: x_transformed shape={x_transformed.shape}")
    
    x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8,random_state=123,shuffle=True
    )

    #2. 모델
    from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
    model = RandomForestClassifier(random_state=123)

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

### 실습 ####
# for문 -> 기본과 차원 세개씩 줄여서 결과 뽑아보기
# 기본결과 :
# 차원 축소 1개 :
# 차원 축소 2개 :
# ....
'''
결과  : 0.9736842105263158
n_components=30: 결과=0.9736842105263158
결과  : 0.9736842105263158
n_components=27: 결과=0.9736842105263158
결과  : 0.9824561403508771
n_components=24: 결과=0.9824561403508771
결과  : 0.9824561403508771
n_components=21: 결과=0.9824561403508771
결과  : 0.9824561403508771
n_components=18: 결과=0.9824561403508771
결과  : 0.9912280701754386
n_components=15: 결과=0.9912280701754386
결과  : 0.9824561403508771
n_components=12: 결과=0.9824561403508771
결과  : 0.9912280701754386
n_components=9: 결과=0.9912280701754386
결과  : 0.9912280701754386
n_components=6: 결과=0.9912280701754386
결과  : 0.956140350877193
n_components=3: 결과=0.956140350877193
최고의 components 수는 {15} 이고 결과는 0.9912280701754386 이다
'''