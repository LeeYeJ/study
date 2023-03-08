from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=650874,train_size=0.9
)
##(442, 10) (442,)

model=Sequential()
model.add(Dense(7, input_dim=10))
model.add(Dense(8))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8))
model.add(Dense(1))

# es=EarlyStopping(monitor='val_loss',patience=100,mode='min',restore_best_weights=True)

es=EarlyStopping(monitor='val_loss', patience=20,mode='min',restore_best_weights=True)

model.compile(loss='mae',optimizer='adam')
hist = model.fit(x_train,y_train,epochs=1000,batch_size=20,validation_split=0.1,callbacks=[es])

loss=model.evaluate(x_test,y_test)
print('loss :', loss)

y_pre=model.predict(x_test)
r2=r2_score(y_pre,y_test)
print('r2:', r2)

plt.plot(hist.history['val_loss'])
plt.plot(hist.history['loss'])

plt.show()

# 얼리 스타핑은 로스값을 계속 비교해주다가 patience 횟수까지 더 나아지지 않으면 정지

'''
Epoch 58/1000
18/18 [==============================] - 0s 2ms/step - loss: 44.0207 - val_loss: 53.9741
Epoch 59/1000
18/18 [==============================] - 0s 2ms/step - loss: 44.0949 - val_loss: 53.9493
2/2 [==============================] - 0s 0s/step - loss: 32.9661
loss : 32.96613311767578
r2: 0.5290313693444784
'''

