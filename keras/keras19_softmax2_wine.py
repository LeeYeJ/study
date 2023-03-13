#사이킷런 로드와인
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np

datasets=load_wine()
x=datasets.data
y=datasets.target

print(y)

print(x.shape,y.shape) # (178, 13) (178,)

print(np.unique(y)) # [0 1 2] # y라벨 추출

#######다중이니까 원핫인코딩해주기###########

y = to_categorical(y)  # 0부터 시작
print(y)
print(y.shape) #(178, 3)

##########################################

x_train,x_test,y_train,y_test=train_test_split(
    x,y, shuffle=True, random_state=3338478, train_size=0.9, stratify=y
)

model=Sequential()
model.add(Dense(10,activation='relu',input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10,activation='relu'))
model.add(Dense(10))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

es=EarlyStopping(monitor='val_loss',mode='auto',patience=20,restore_best_weights=True)

model.fit(x_train,y_train,epochs=500,batch_size=1,validation_split=0.1,callbacks=[es])

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_pre=np.argmax(model.predict(x_test),axis=1)


y_pre = to_categorical(y_pre)
# print(y_pre)
# print(y_pre.shape) #(178, 3)


# y_pre=np.round(model.predict(x_test))
print(y_pre)
print("============")
print(y_test)

acc=accuracy_score(y_test,y_pre)
print('acc :', acc)

'''
x_train,x_test,y_train,y_test=train_test_split(
    x,y, shuffle=True, random_state=3338478, train_size=0.9, stratify=y
)

Epoch 52/500
144/144 [==============================] - 0s 977us/step - loss: 0.2069 - acc: 0.9167 - val_loss: 0.4883 - val_acc: 0.8125
Epoch 53/500
144/144 [==============================] - 0s 943us/step - loss: 0.2312 - acc: 0.8958 - val_loss: 0.3871 - val_acc: 0.8750
1/1 [==============================] - 0s 103ms/step - loss: 0.2214 - acc: 0.9444
loss : [0.22139905393123627, 0.9444444179534912]
acc : 0.9444444444444444
'''






