import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape input data
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Normalize input data
x_train = x_train / 255.0
x_test = x_test / 255.0

n_components = [154, 331, 486, 713]
accuracys =[]
for i in n_components:
    # Apply PCA
    pca = PCA(n_components=i)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # Convert labels to one-hot encoding
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    # Define DNN model
    model = Sequential()
    model.add(Dense(128, input_shape=(i,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train DNN model
    model.fit(x_train_pca, y_train_one_hot, epochs=5, batch_size=32)

    # Evaluate DNN model
    _, accuracy = model.evaluate(x_test_pca, y_test_one_hot, verbose=0)
    accuracys.append(accuracy)
    print(f'PCA n_components={i}, Accuracy: {accuracy:.3f}')
print(accuracys)  

'''
PCA n_components=154, Accuracy: 0.975
PCA n_components=331, Accuracy: 0.974
PCA n_components=486, Accuracy: 0.974
PCA n_components=713, Accuracy: 0.971
[0.9746999740600586, 0.9740999937057495, 0.973800003528595, 0.9714999794960022]
'''  
    # result = model.evaluate(x_test,y_test)
    # y_pred = model.predict(x_test)
    # y_pred = np.argmax(y_pred, axis=1)
    # y_test = np.argmax(y_test, axis=1)
    # acc = accuracy_score(y_pred,y_test)
    
    # results.append(acc)
    # print('PCA 가',i,'acc :',acc)





###########################################################################################################







# from tensorflow.keras.datasets import mnist
# import numpy as np
# from sklearn.decomposition import PCA
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Conv2D,Flatten
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score


# (x_train, y_train ),(x_test , y_test )= mnist.load_data() # 둘다 뽑기 싫으면 _ 해주면 됨 (파이썬 기초 문법) -> 특수문자가 변수로 먹힘(메모리 할당됨)
# print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

# # print( _.shape) # (10000,)
# # print( __.shape) #(60000,)

# # 이미지 데이터를 피면 (dnn을 사용해줄때) 각각이 컬럼이 된다 (성능이 많이 차이 안남)

# # 합쳐줄때 둘다 가능
# # x = np.concatenate((x_train,x_test), axis=0) #(70000, 28, 28)
# # x = np.append(x_train,x_test,axis=0) #(70000, 28, 28)
# # print(x.shape) 

# # 스케일러처럼 PCA는 2차원만 받을수있다. 
# # x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
# # print(x.shape) #(70000, 784)

# ############ 실습 ###############

# # pca를 통해 0.95 이상인 n_component는 몇개?
# # 0.95 몇개 ??
# # 0.99 몇개 ??
# # 0.999 몇개 ??
# # 1.0 몇개 ??
# # 힌트 -> np.argmax

# print('나의 최고의 CNN :')
# print('나의 최고의 DNN :')


# n_components = [154,331,486,713]
# results = []
# for i in n_components:
#     # x = np.append(x_train,x_test,axis=0)  # 컬럼 값이 계속 바뀌니까 데이터를 for문 안에서 처리해줘야됨
#     x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
#     x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])
    
    
    
#     pca = PCA(n_components=int(i)) # 리스트 안에 있으니까 문자형으로 받아서 int형으로 바꿔줘야됨
#     x_train = pca.fit_transform(x_train) 
#     x_test = pca.fit_transform(x_test) 
    
#     print(x_train.shape,x_test.shape)
    
#     y_train = to_categorical(y_train)
#     y_test = to_categorical(y_test)
    
#     # x_train= x_train.reshape(60000,i)
#     # x_test= x_test.reshape(10000,i)
       
#     model= Sequential()
#     model.add(Dense(8,input_shape=(i,))) # 28*28로 표현해줘도됨
#     model.add(Dense(12, activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(9,))
#     model.add(Dense(7,))
#     model.add(Dense(10,activation='softmax'))

#     model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
    
#     model.fit(x_train,y_train, epochs =10, batch_size = 30)
    
#     result = model.evaluate(x_test,y_test)
#     y_pred = model.predict(x_test)
#     y_pred = np.argmax(y_pred, axis=1)
#     y_test = np.argmax(y_test, axis=1)
#     acc = accuracy_score(y_pred,y_test)
    
#     results.append(acc)
#     print('PCA 가',i,'acc :',acc)
    
#     # print('PCA 가',i,'acc :',result)

# print(results)
    
# '''

# '''    

# # pca_EVR = pca.explained_variance_ratio_

# # cumsum = np.cumsum(pca_EVR)
# # print(cumsum) 

# # print(np.argmax(cumsum >= 0.95)+1) # 712 -> 0부터 시작하니까 713개이다. 따라서 +1해줌
# # print(np.argmax(cumsum >= 0.99)+1) # 712 -> 0부터 시작하니까 713개이다. 따라서 +1해줌
# # print(np.argmax(cumsum >= 0.999)+1) # 712 -> 0부터 시작하니까 713개이다. 따라서 +1해줌
# # print(np.argmax(cumsum >= 1.0)+1) # 712 -> 0부터 시작하니까 713개이다. 따라서 +1해줌

# ####################### 실습 #########################
# # 모델 만들어서 비교
# #                        acc값
# # 1. 나의 최고의 CNN :
# # 2. 나의 최고의 DNN :
# # 3. PCA 0.95       : 
# # 3. PCA 0.99       :
# # 3. PCA 0.999      :
# # 3. PCA 1.0        :




