#pca - 차원(컬럼) 축소(압축)하는 개념
# 쓸모없는 컬럼들은 압축되면 성능이 좋아질수도있다
# 차원을 축소해서 생긴 y값 -> 타겟값이 생기니까 비지도이다. / -> 스케일러의 개념도?

# np.cumsum으로 최대 차원축소율 확인 가능 (참고용)
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

pca = PCA(n_components=30)
x = pca.fit_transform(x)
print(x.shape)

pca_EVR =pca.explained_variance_ratio_ # explained_variance_ratio_ -> 설명가능한 변화율
print(pca_EVR)
print(sum(pca_EVR)) # 합이 0.9999999999999998

pca_cumsum=np.cumsum(pca_EVR)
print(np.cumsum(pca_EVR)) # 누적합 ( 원본과의 일치율 )
'''
[0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
 0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
 0.99999999 0.99999999 1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.        ]
 
위 pca가 15번째에 1이 되었을때 원본과 일치된다. (원본 손상 X)
'''
#누적합에 대한 그림
import matplotlib.pyplot as plt
plt.plot(pca_cumsum)
plt.grid()
plt.show()
