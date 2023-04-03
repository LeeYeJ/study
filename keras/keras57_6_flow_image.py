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

# 실습
# x_augment 10개와 x_train 10개를 비교하는 이미지 출력

import matplotlib.pylab as plt
plt.figure(figsize=(2,10))# figsize 그림그릴 판
for i in range(9):
    plt.subplot(2,10,i+1)
    plt.axis('off')
    # plt.imshow(x_data[0][0][i],cmap = 'gray') # x_data[0] 배치크기 포함 / .next()미사용
    plt.imshow(x_train[0][0][i], cmap='gray') # .next() 사용
plt.show()