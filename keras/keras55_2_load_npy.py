
#반전시키면 알아볼수있음? / 바이너리 -> 수치화/ numpy -> unique = padas -> value_counts

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path ='d:/study_data/_save/_npy/'
# np.save(path+'keras55_1_x_train.npy',arr =xy_train[0][0]) #xy_train[0][0]이 저 경로에 저장됨
# np.save(path+'keras55_1_x_test.npy',arr =xy_train[0][0])
# np.save(path+'keras55_1_y_train.npy',arr =xy_train[0][1])
# np.save(path+'keras55_1_y_test.npy',arr =xy_train[0][1])

x_train = np.load(path+'keras55_1_x_train.npy')
x_test = np.load(path+'keras55_1_x_test.npy')
y_train = np.load(path+'keras55_1_y_train.npy')
y_test = np.load(path+'keras55_1_y_test.npy')


print(x_train)
print(x_train.shape)
print(x_test)
print(x_test.shape)
print(y_train)
print(y_train.shape)
print(y_test)
print(y_test.shape)

'''
x와 y가 합쳐진 iterator형태는 같이 넣어도 됨
batch_size=10일 때  
print(xy_train[0][0])        #x 10개 들어가있음
print(xy_train[0][1])        #[0. 1. 0. 1. 1. 0. 1. 1. 1. 0.]
print(xy_train[0][0].shape)  #(10, 200, 200, 1)
print(xy_train[0][1].shape)  #(10,)
'''
'''
# print("===========================================================")
# print(type(xy_train))  #<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(type(xy_train[0][1])) #<class 'numpy.ndarray'>
# '''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(100,100,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(15))
model.add(Dense(20,activation='relu'))
model.add(Dense(15))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics=['acc'])

# model.fit(xy_train[0][0],xy_train[0][1])  # 얘는 통으로 들어감
# hist = model.fit_generator(xy_train,
#                     epochs=100, # fit generator는 배치 사이즈까지 처리해서 한번에 훈련
#                     # steps_per_epoch=32,  # 전체 데이터에서 배치 나눈 값으로 주면 됨
                    # validation_data=xy_test,
#                     validation_steps=24) # 전체 데이터 나누기 배치

hist = model.fit(x_train,y_train,
                    epochs=100, # fit generator는 배치 사이즈까지 처리해서 한번에 훈련
                    steps_per_epoch=32,  # 전체 데이터에서 배치 나눈 값으로 주면 됨
                    validation_data=(x_test,y_test),
                    validation_steps=24) # 전체 데이터 나누기 배치

from sklearn.metrics import accuracy_score

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss :',loss[-1])



