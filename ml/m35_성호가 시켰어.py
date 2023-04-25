# 컬럼의 갯수가 클래스의 갯수보다 작을때 디폴트로 돌아가냐 확인


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #->LDA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris,load_breast_cancer,load_digits 
from tensorflow.keras.datasets import cifar100

#데이터

(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape) #(n, 32,32,3)

x_train = x_train.reshape(50000,32*32*3)


pca = PCA(n_components=98)
x_train = pca.fit_transform(x_train)
# print(x.shape) 

lda = LinearDiscriminantAnalysis() # 클래스의 위치 표시 / 디폴트 => 클래스 -1 or n_feature에서 최소값이 나옴 ( 즉 여기선 디폴트 99보다 줄여준 98이 더 작으니까 98로 나옴)
# n_components는 클래스의 갯수 빼기 하나 이하로 가능하다!! 
x = lda.fit_transform(x_train,y_train) 
print(x.shape) #(50000, 98)


