# 6만개의 데이터에서 4만개 뽑아서 10만개로 증폭

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
np.random.seed(333)

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 40000 #argument_size증폭 / 
# 6만개에서 4만개를 뽑음
# randinedx = np.random.randint(60000,size = 40000) 
randinedx = np.random.randint(x_train.shape[0], size=augment_size) # x_train.shape (60000,28,28)
print(randinedx) #[35868 44983 27953 ... 14127 19037 44790] 랜덤값
print(randinedx.shape) #(40000,)
print(np.min(randinedx),np.max(randinedx)) # 랜덤값이니까 최소최대값이 바뀔수있음

x_augmented = x_train[randinedx].copy() # 4만개의 데이터가 x_augmented에 들어간다
y_augmented = y_train[randinedx].copy() # 원데이터를 해치지 않기 위해 .copy()를 써줌 
print(x_augmented)
print(x_augmented.shape, y_augmented.shape) #(40000, 28, 28) (40000,)

x_train = x_train.reshape(60000,28,28,1)

x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],
                        1)

x_augmented = x_augmented.reshape(
                        x_augmented.shape[0],
                        x_augmented.shape[1],
                        x_augmented.shape[2],
                        1)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented, batch_size=augment_size, shuffle=False
) .next()[0] #.next()[0] = x_augmented[0][0]
# print(x_augmented) #<keras.preprocessing.image.NumpyArrayIterator object at 0x0000025591263C40>

# print(x_augmented[0][0].shape) #  (40000, 28, 28) (40000,)

print(x_augmented)
print(x_augmented.shape) #(40000, 28, 28, 1)

print(np.max(x_train), np.min(x_train)) # x_augmented는 스케일러 되어있는데 x_train은 안되어있음 그래서 x_train따로 스케일러 해줌
print(np.max(x_augmented), np.min(x_augmented))

x_train  =  np.concatenate((x_train/255.,x_augmented))
y_train = np.concatenate((y_train, y_augmented))
x_test = x_test/255.

print(np.max(x_train), np.min(x_train)) # x_augmented는 스케일러 되어있는데 x_train은 안되어있음 그래서 x_train따로 스케일러 해줌
print(np.max(x_augmented), np.min(x_augmented))

print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)
print(x_test.shape, y_test.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(type(x_augmented))



# 모델 만들어보자 # 증폭 성능 비교
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten

#2.모델 구성
model= Sequential()
model.add(Conv2D(8,2,input_shape=(28,28,1),activation='linear')) # 28*28로 표현해줘도됨
model.add(Flatten())
model.add(Dense(12, activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(10,activation='softmax'))

# 소프트맥스는 라벨값들의 확률이 나오기 때문에 마지막에 알그맥스로 변환해준다.
import time
start_time=time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

# import datetime # 시간을 저장해줌
# date = datetime.datetime.now() # 현재 시간
# print(date) # 2023-03-14 11:15:39.585470
# date = date.strftime('%m%d_%H%M') # 시간을 문자로 바꾼다 ( 월, 일, 시 ,분)
# print(date) # 0314_1115

# filepath='./_save/MCP/keras27_4/'
# filename = '{epoch:04d}-{val_loss:4f}.hdf5' #val_loss:4f 소수 넷째자리까지 받아와라


from tensorflow.python.keras.callbacks import EarlyStopping

es=EarlyStopping(monitor='loss',mode='auto',patience=10)

# mcp= ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # val_loss 기준, verbose=1 훈련중 확인 가능
#                     save_best_only=True,  # 가장 좋은 지점에서 세이브하기
#                     filepath="".join([filepath, 'k27_', date,'_',filename ])) # 경로는 이곳에 / .join 합친다는 뜻


hist = model.fit(x_train,y_train,epochs=1, batch_size=512,validation_split=0.1, callbacks=[es])
end_time = time.time()

results= model.evaluate(x_test,y_test)
print('results :', results)



y_pred=model.predict(x_test)
# print(type(y_pred))
# y_test_acc=np.argmax(y_test) 
# print(y_test_acc) 
# y_pred=np.argmax(y_pred)
# print(y_pre) 



from sklearn.metrics import accuracy_score
acc=accuracy_score(y_pred,y_test)   
print('acc :', acc)

print('time:', round(end_time-start_time,2))