#사이킷런 로드 디짓


#사이킷런 로드와인
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np

datasets=load_digits()
x=datasets.data
y=datasets.target

print(y)

print(x.shape,y.shape) # (1797, 64) (1797,)

print(np.unique(y)) # [0 1 2 3 4 5 6 7 8 9] # y라벨 추출

#######다중이니까 원핫인코딩해주기###########

y = to_categorical(y)
print(y) 
print(y.shape) #(1797, 10)

##########################################

x_train,x_test,y_train,y_test=train_test_split(
    x,y, shuffle=True, random_state=3338478, train_size=0.9, stratify=y
)

model=Sequential()
model.add(Dense(10,activation='relu',input_dim=64))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10,activation='relu'))
model.add(Dense(10))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

es=EarlyStopping(monitor='val_loss',mode='auto',patience=20,restore_best_weights=True)

model.fit(x_train,y_train,epochs=500,batch_size=50,validation_split=0.1,callbacks=[es])

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_pre=np.round(model.predict(x_test))
acc=accuracy_score(y_test,y_pre)
print('acc :', acc)

'''


x_train,x_test,y_train,y_test=train_test_split(
    x,y, shuffle=True, random_state=3338478, train_size=0.9, stratify=y
)

30/30 [==============================] - 0s 2ms/step - loss: 0.0753 - acc: 0.9787 - val_loss: 0.3818 - val_acc: 0.9136
Epoch 77/500
30/30 [==============================] - 0s 2ms/step - loss: 0.0771 - acc: 0.9794 - val_loss: 0.3929 - val_acc: 0.9012
6/6 [==============================] - 0s 845us/step - loss: 0.2289 - acc: 0.9222
loss : [0.2289174646139145, 0.9222221970558167]
acc : 0.9111111111111111
'''






