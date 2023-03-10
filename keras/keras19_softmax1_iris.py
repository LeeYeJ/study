#다중분류
#줄기와 잎으로 꽃품종을 맞춘다.
#다중분류는 무조건 softmax
#argmax를 사용해서 y_test, y_pre의 shape를 맞춰준다.

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터
datasets=load_iris()
print(datasets.DESCR)   #pandas : descibe()
'''
        - class: # 라벨값
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica

'''
print(datasets.feature_names)   ##pandas :.columns()

x=datasets.data
y=datasets['target']
print(x.shape,y.shape) #(150, 4) (150,)
print(x)
print(y)

print('y의 라벨값:',np.unique(y))  # y의 라벨을 확인하려면 -> np.unique(y)

##################여기 지점에서 원핫을 해줘야된다.###################

# ## y를 (150,1) ->(150,3)으로
# y_data = tf.keras.utils.to_categorical(y, num_classes=3)
# print(y_data.shape) #(150, 3)

#####교수님이랑 한 것. 원핫 인코딩####################
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
print(y.shape) #(150, 3)
################ 판다스 겟더미########################
# num = np.unique(y, axis=0)
# print(num)
# num = num.shape[0]

# encoding = np.eye(num)[y]
###########사이킷런에 원핫인코더#################
print(y)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=333,train_size=0.9,stratify=y
    
    # stratify=y 각 라벨의 값들을 일정 비율로 뽑아준다.
) 
print(y_train)
print(np.unique(y_train, return_counts=True)) #return_counts=True 갯수까지 반환해준다.

#모델구성
model = Sequential()
model.add(Dense(50,activation='relu',input_dim=4))
model.add(Dense(40,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation='softmax')) # softmax -> y밸류 3가지가 아웃풋 노드값이 되고 확률분배까지가 모델에서 이뤄진다.

#accuracy_score를 사용헤서 스코어를 빼세요.
#numpy에 있는 무언가를 이용해서 제일 큰값을 골라 1 ->argmax()

# 평가 예측

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

es=EarlyStopping(monitor='val_accuracy', mode='auto',patience=30,restore_best_weights=True)

model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2,verbose=1,callbacks=[es])

results=model.evaluate(x_test,y_test)
print('results:',results)
print('loss:', results[0])
print('acc:',results[1])
'''
results: [0.2570165991783142, 0.9333333373069763]
loss: 0.2570165991783142
acc: 0.9333333373069763
'''

y_pre= model.predict(x_test) # acc만들라고 만든거임
print(y_pre.shape)
print(y_pre[:5])
'''
(15, 3)
[[1.0000000e+00 1.4443072e-08 5.6063612e-11]
 [1.0000000e+00 4.6158274e-08 2.5063840e-10]
 [1.0000000e+00 1.4543861e-08 5.2665785e-11]
 [6.2764212e-07 6.8659335e-02 9.3134010e-01]
 [9.2028822e-11 4.4800108e-03 9.9552000e-01]]
'''
print("============")
print(y_test.shape) #원핫이 되어있음  #현재는 소프트맥스의 결과
print(y_test[:5])
'''
(15, 3)
[[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 0. 1.]]
'''
####################################
# argmax를 사용을 y_pre, y_test 둘다 해준다. 그러면 (최대값의 위치값을 빼주니까!)

y_test_acc=np.argmax(y_test, axis=1) # 각 행에 있는 열끼리 비교
print(y_test_acc) # [0 0 0 2 2 1 2 0 2 0 1 1 1 2 1]
y_pre=np.argmax(y_pre,axis=1)
print(y_pre) # [0 0 0 2 2 1 2 0 2 0 1 2 1 2 1]

acc=accuracy_score(y_pre,y_test_acc)
print('acc:', acc)




