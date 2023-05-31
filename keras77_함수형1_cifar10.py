# 함수형 맹그러바~

# 가중치 동결과 하지 않았을때 , 그리고 원래와 성능 비교
# Flatten과 GAP과 차이

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Flatten,Dropout,Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping

# 데이터
(x_train,y_train),(x_test,y_test) =cifar10.load_data()

print(x_train.shape,y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape) # (10000, 32, 32, 3) (10000, 1)

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)  

print(y_train.shape) 
print(y_test.shape)  

x_train = x_train/255.
x_test = x_test/255.
print(x_train.shape,x_test.shape)

# model = VGG16()  # include_top=True, input_shape=(224,224,3)
vgg16  = VGG16(weights='imagenet', include_top=False, # 완전 연결 계층(top layer)을 포함할지 여부
               input_shape=(32,32,3))

vgg16.trainable = False # 가중치 동결 ( 통상 동결 시키고 커스터마이징 한 부분만 훈련하는것이 좋지만 모름 해봐야알아 )
    # 30
    # 4 -> 마지막 Dense

    
# 모델 입력
input_tensor = Input(shape=(32,32,3))

# VGG16 모델의 출력을 입력으로 받는 새로운 레이어 구성
x = vgg16(input_tensor)
dense2 = Flatten()(x)
dense3 = Dense(64)(dense2)
dense4 = Dense(32,activation='relu')(dense3)
dense5 = Dense(35)(dense4)
drop2 = Dropout(0.2)(dense5)
output1 = Dense(10, activation='softmax')(drop2)
model = Model(inputs=input_tensor, outputs=output1)

# model.trainable = True

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es=EarlyStopping(monitor='loss',mode='auto',patience=10)

model.fit(x_train,y_train,epochs=1, batch_size=512,validation_split=0.1, callbacks=[es])

#4. 평가, 예측 
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)

y_test_acc=np.argmax(y_test,axis=1) 
# print(y_test_acc) 
y_pred=np.argmax(y_predict,axis=1)
# print(y_pre) 

acc=accuracy_score(y_pred,y_test_acc)   
print('model 결과 :', results)
print('acc :', acc)

# 원래
# Epoch 50/50
# 88/88 [==============================] - 12s 137ms/step - loss: 1.2666 - val_loss: 1.2372
# 313/313 [==============================] - 1s 3ms/step - loss: 1.2406
# results : 1.2405617237091064
# acc : 0.5574


# trainable = False ( 동결 ) - flatten
# model 결과 : 0.055167827755212784
# acc : 0.5858

# trainable = False ( 동결 ) - GAP
# model 결과 : 0.05518114194273949
# acc : 0.5852


# trainable = False ( 동결 x ) - flatten
# model 결과 : 0.17999997735023499
# acc : 0.1

# trainable = False ( 동결 x ) - GAP
# model 결과 : 0.18000000715255737
# acc : 0.1




