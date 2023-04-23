from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA

(x_train, __ ),(x_test , _ )= mnist.load_data() # 둘다 뽑기 싫으면 _ 해주면 됨 (파이썬 기초 문법) -> 특수문자가 변수로 먹힘(메모리 할당됨)

print( _.shape) # (10000,)
print( __.shape) #(60000,)

# 이미지 데이터를 피면 (dnn을 사용해줄때) 각각이 컬럼이 된다 (성능이 많이 차이 안남)

# 합쳐줄때 둘다 가능
# x = np.concatenate((x_train,x_test), axis=0) #(70000, 28, 28)
x = np.append(x_train,x_test,axis=0) #(70000, 28, 28)
print(x.shape) 

# 스케일러처럼 PCA는 2차원만 받을수있다. 
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
print(x.shape) #(70000, 784)

############ 실습 ###############

# pca를 통해 0.95 이상인 n_component는 몇개?
# 0.95 몇개 ??
# 0.99 몇개 ??
# 0.999 몇개 ??
# 1.0 몇개 ??
# 힌트 -> np.argmax

pca = PCA(n_components=784)
x = pca.fit_transform(x)
print(x) 

pca_EVR = pca.explained_variance_ratio_

cumsum = np.cumsum(pca_EVR)
print(cumsum) 

print(np.argmax(cumsum >= 0.95)+1) # 712 -> 0부터 시작하니까 713개이다. 따라서 +1해줌
print(np.argmax(cumsum >= 0.99)+1) # 712 -> 0부터 시작하니까 713개이다. 따라서 +1해줌
print(np.argmax(cumsum >= 0.999)+1) # 712 -> 0부터 시작하니까 713개이다. 따라서 +1해줌
print(np.argmax(cumsum >= 1.0)+1) # 712 -> 0부터 시작하니까 713개이다. 따라서 +1해줌
