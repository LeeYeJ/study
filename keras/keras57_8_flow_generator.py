# 57_5 카피해서 복붙

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

train_datagen2 = ImageDataGenerator(
    # rescale=1./1,
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

x_augmented = train_datagen.flow( # 이터레이터 형태로 뽑음
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

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(np.max(x_train), np.min(x_train)) # x_augmented는 스케일러 되어있는데 x_train은 안되어있음 그래서 x_train따로 스케일러 해줌
# print(np.max(x_augmented), np.min(x_augmented))

###################### x,y 합치기 #########################
batch_size=6
xy_train = train_datagen2.flow(x_train,y_train,batch_size=batch_size, shuffle=True)

#2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(64,2,input_shape=(28,28,1)))
model.add(Conv2D(64,2))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

#3. 컴파일
model.compile (loss ='categorical_crossentropy', optimizer='adam',metrics=['acc'])

model.fit_generator(xy_train, epochs = 1,
                    steps_per_epoch=len(xy_train)/batch_size)


print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)


#오류남
# x_train = x_train + x_augmented
# print(x_train.shape) 

# print(x_train.shape) #(60000, 28, 28)
# print(x_train[0].shape) #(28, 28) -> 
# print(x_train[1].shape) #(28, 28) -> 
# print(x_train[0][0].shape) #(28,) -> 

# print(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1).shape) #(100, 28, 28, 1)
# #np.tile 데이터,증폭시킬 갯수

# print(np.zeros(augment_size)) # 백개의 0을 출력
# print(np.zeros(augment_size).shape) #(100,)

# x_data = train_datagen.flow( # 원 데이터를 증폭이나 변환
#     np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1), # x데이터 #경로가 아닌 데이터를 넣어준다.
#     np.zeros(augment_size), # y데이터 : 그림만 그릴거라 필요없어서 걍 0 넣어줘
#     batch_size = augment_size,
#     shuffle = True
# )

# print(x_data)

# print(x_data[0]) # x y가 모두 포함
# print(x_data[0][0].shape) #(100, 28, 28, 1)
# print(x_data[0][1].shape) #(100,)

# augment_size = 100 # 변환 데이터는 100개 중 49개

# import matplotlib.pylab as plt
# plt.figure(figsize=(7,7))# figsize 그림그릴 판
# for i in range(49):
#     plt.subplot(7,7,i+1)
#     plt.axis('off')
#     plt.imshow(x_data[0][0][i],cmap = 'gray')
# plt.show()