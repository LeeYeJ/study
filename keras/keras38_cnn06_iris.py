#다중분류
#줄기와 잎으로 꽃품종을 맞춘다.
#다중분류는 무조건 softmax
#argmax를 사용해서 y_test, y_pre의 shape를 맞춰준다.

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Conv2D,Flatten
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

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

print(x_train.shape) #(135, 4)
print(x_test.shape) #(15, 4)

print(y_train)
print(np.unique(y_train, return_counts=True)) #return_counts=True 갯수까지 반환해준다.

scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
# scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
scaler.fit(x_test)
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

# #모델구성
# model = Sequential()
# model.add(Dense(50,activation='relu',input_dim=4))
# model.add(Dense(40,activation='relu'))
# model.add(Dense(40,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(3,activation='softmax')) # softmax -> y밸류 3가지가 아웃풋 노드값이 되고 확률분배까지가 모델에서 이뤄진다.

x_train= x_train.reshape(135,2,2,1)
x_test= x_test.reshape(15,2,2,1)

model = Sequential()
model.add(Conv2D(7,(2,1),input_shape=(2,2,1)))
model.add(Conv2D(8,(2,1),activation='relu',padding='same'))
model.add(Conv2D(5,(2,1),padding='same'))
model.add(Flatten())
model.add(Dense(9,activation='relu'))
model.add(Dense(6))
model.add(Dense(3,activation='softmax'))

# input1 = Input(shape=(4,))
# modell = Dense(50, activation='relu')(input1)
# model2 = Dense(40, activation='linear')(modell)
# model3 = Dense(40, activation='linear')(model2)
# model4 = Dense(10, activation='linear')(model3)
# output1 = Dense(3, activation='softmax')(model4)
# model= Model(inputs=input1,outputs=output1)

#accuracy_score를 사용헤서 스코어를 빼세요.
#numpy에 있는 무언가를 이용해서 제일 큰값을 골라 1 ->argmax()

# 평가 예측

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

es=EarlyStopping(monitor='val_acc', mode='auto',patience=30,restore_best_weights=True)

model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2,verbose=1,callbacks=[es])

results=model.evaluate(x_test,y_test)
print('results:',results)
print('loss:', results[0])
print('acc:',results[1])

y_pre= model.predict(x_test) # acc만들라고 만든거임
# print(y_pre.shape)
# print(y_pre[:5])

# print("============")
# print(y_test.shape) #원핫이 되어있음  #현재는 소프트맥스의 결과
# print(y_test[:5])

####################################
# argmax를 사용을 y_pre, y_test 둘다 해준다. 그러면 (최대값의 위치값을 빼주니까!)

y_test_acc=np.argmax(y_test, axis=1) # 각 행에 있는 열끼리 비교
# print(y_test_acc) # [0 0 0 2 2 1 2 0 2 0 1 1 1 2 1]
y_pre=np.argmax(y_pre,axis=1)
# print(y_pre) # [0 0 0 2 2 1 2 0 2 0 1 2 1 2 1]

acc=accuracy_score(y_pre,y_test_acc)
print('acc:', acc)

'''
MinMaxScaler

108/108 [==============================] - 0s 1ms/step - loss: 0.0138 - acc: 0.9907 - val_loss: 0.0551 - val_acc: 0.9630
1/1 [==============================] - 0s 108ms/step - loss: 0.5934 - acc: 0.6667
results: [0.5934094786643982, 0.6666666865348816]
loss: 0.5934094786643982
acc: 0.6666666865348816
acc: 0.6666666666666666

StandardScaler

108/108 [==============================] - 0s 991us/step - loss: 0.0200 - acc: 0.9815 - val_loss: 0.1309 - val_acc: 0.9630
1/1 [==============================] - 0s 102ms/step - loss: 0.7256 - acc: 0.7333
results: [0.7256137132644653, 0.7333333492279053]
loss: 0.7256137132644653
acc: 0.7333333492279053
acc: 0.7333333333333333

RobustScaler

108/108 [==============================] - 0s 1ms/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.1125 - val_acc: 0.9630
1/1 [==============================] - 0s 99ms/step - loss: 0.3298 - acc: 0.8000
results: [0.3298465311527252, 0.800000011920929]
loss: 0.3298465311527252
acc: 0.800000011920929
acc: 0.8

MaxAbsScaler

Epoch 33/100
108/108 [==============================] - 0s 994us/step - loss: 0.0645 - acc: 0.9722 - val_loss: 0.0326 - val_acc: 1.0000
1/1 [==============================] - 0s 104ms/step - loss: 0.2512 - acc: 0.8667
results: [0.2511949837207794, 0.8666666746139526]
loss: 0.2511949837207794
acc: 0.8666666746139526
acc: 0.8666666666666667


'''
'''
MinMaxScaler 

Epoch 35/100
108/108 [==============================] - 0s 1ms/step - loss: 0.0999 - acc: 0.9537 - val_loss: 0.0259 - val_acc: 1.0000
1/1 [==============================] - 0s 98ms/step - loss: 0.2292 - acc: 0.8000
results: [0.22920073568820953, 0.800000011920929]
loss: 0.22920073568820953
acc: 0.800000011920929
acc: 0.8

StandardScaler 

Epoch 39/100
108/108 [==============================] - 0s 1ms/step - loss: 0.0471 - acc: 0.9815 - val_loss: 0.0978 - val_acc: 0.9630
1/1 [==============================] - 0s 113ms/step - loss: 0.3757 - acc: 0.8000
results: [0.37572115659713745, 0.800000011920929]
loss: 0.37572115659713745
acc: 0.800000011920929
acc: 0.8

RobustScaler

Epoch 32/100
108/108 [==============================] - 0s 1ms/step - loss: 0.0152 - acc: 0.9907 - val_loss: 0.0382 - val_acc: 0.9630
1/1 [==============================] - 0s 112ms/step - loss: 0.3886 - acc: 0.8000
results: [0.3885835111141205, 0.800000011920929]
loss: 0.3885835111141205
acc: 0.800000011920929
acc: 0.8

MaxAbsScaler

Epoch 31/100
108/108 [==============================] - 0s 964us/step - loss: 0.0922 - acc: 0.9722 - val_loss: 0.0246 - val_acc: 1.0000
1/1 [==============================] - 0s 103ms/step - loss: 0.8809 - acc: 0.8667
results: [0.8808686137199402, 0.8666666746139526]
loss: 0.8808686137199402
acc: 0.8666666746139526
acc: 0.8666666666666667

CNN모델
Epoch 37/100
108/108 [==============================] - 0s 1ms/step - loss: 0.0486 - acc: 0.9815 - val_loss: 0.0192 - val_acc: 1.0000
1/1 [==============================] - 0s 112ms/step - loss: 0.2916 - acc: 0.8000
results: [0.2915595471858978, 0.800000011920929]
loss: 0.2915595471858978
acc: 0.800000011920929
acc: 0.8

'''


