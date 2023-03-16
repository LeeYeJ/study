from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping 
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score


# from sk
import numpy as np

#1데이터
(x_train,y_train),(x_test,y_test)=mnist.load_data() # 이 순서대로 값을 던져준다

#################################################
#실습

x_train = x_train.reshape(60000,28,28,1) # 구조만 달라지고 순서와 값은 바뀌지 않는다.
x_test = x_test.reshape(10000,28,28,1)

#스케일링 이렇게 해줘도됨
x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train),np.min(x_train))

#혹은 이렇게 리쉐잎과 스케일링 동시에
# x_train= x_train.reshape(60000,28.28,1)/255.
# x_test= x_test.reshape(60000,28.28,1)/255.


print(np.unique(y_train,return_counts=True)) #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

print(y_train) #[5 0 4 ... 5 6 8]
print(y_train.shape)
print(y_test)  #[7 2 1 ... 4 5 6]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)
print(y_test.shape) 
 
# print(y.shape)


x_train = x_train.reshape(60000,28*28) # 구조만 달라지고 순서와 값은 바뀌지 않는다.
x_test = x_test.reshape(10000,28*28)


Scaler= MinMaxScaler()
Scaler.fit(x_train)
x_train = Scaler.transform(x_train)
x_test = Scaler.transform(x_test)

x_train = x_train.reshape(60000,28,28,1) # 구조만 달라지고 순서와 값은 바뀌지 않는다.
x_test = x_test.reshape(10000,28,28,1)

model = Sequential()
model.add(Conv2D(7,(2,2),input_shape=(28,28,1)))
model.add(Conv2D(8,(3,3),activation='relu'))
model.add(Conv2D(5,(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(9,activation='relu'))
model.add(Dense(6))
model.add(Dense(10,activation='softmax'))

# 소프트맥스는 라벨값들의 확률이 나오기 때문에 마지막에 알그맥스로 변환해준다.

model.compile(loss='categorical_crossentropy', optimizer='adam')
es=EarlyStopping(monitor='loss',mode='auto',patience=50)
model.fit(x_train,y_train,epochs=1, batch_size=5000,validation_split=0.2, callbacks=[es])

results= model.evaluate(x_test,y_test)
print('results :', results)

y_pred=np.argmax(model.predict(x_test),axis=1)
y_test_acc = np.argmax(x_test, axis=1)

acc= accuracy_score(y_test_acc, y_pred)
print('acc:', acc)


#맥스 풀링 -> 속도와 성능이 좋다
#커널 사이즈에서 가장 큰 값만을 추출해 (배경은 무시해줘도 괜찮다 치고)
#0*0도 연산이니까 안하는게 낫다
