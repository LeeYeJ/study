import numpy as np

save_path = 'd:/study_data/_save/Breed/'

x_train = np.load(save_path+'keras56_5_x_train.npy')
x_test = np.load(save_path+'keras56_5_x_test.npy')
y_train = np.load(save_path+'keras56_5_y_train.npy')
y_test = np.load(save_path+'keras56_5_y_test.npy')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten,Dropout

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(150,150,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(15))
model.add(Dense(32))
model.add(Dense(54))
model.add(Dropout(0.3))
model.add(Dense(20,activation='relu'))
model.add(Dense(15))
model.add(Dense(5,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics=['acc'])

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
# val_loss = hist.history['val_loss']
acc = hist.history['acc']
# val_acc = hist.history['val_acc']

print('loss :',loss[-1])
print('acc :',acc[-1])

'''
loss : 0.0005217859870754182
acc : 1.0
'''

