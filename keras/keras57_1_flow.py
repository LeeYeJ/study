from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

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

augment_size = 100 #argument_size증폭

print(x_train.shape) #(60000, 28, 28)
print(x_train[0].shape) #(28, 28) -> 
print(x_train[1].shape) #(28, 28) -> 
print(x_train[0][0].shape) #(28,) -> 

print(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1).shape) #(100, 28, 28, 1)
#np.tile 데이터,증폭시킬 갯수

print(np.zeros(augment_size)) # 백개의 0을 출력
print(np.zeros(augment_size).shape) #(100,)

x_data = train_datagen.flow( # 원 데이터를 증폭이나 변환
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1), # x데이터 #경로가 아닌 데이터를 넣어준다.
    np.zeros(augment_size), # y데이터 : 그림만 그릴거라 필요없어서 걍 0 넣어줘
    batch_size = augment_size,
    shuffle = True
)

print(x_data)

print(x_data[0]) # x y가 모두 포함
print(x_data[0][0].shape) #(100, 28, 28, 1)
print(x_data[0][1].shape) #(100,)

import matplotlib.pylab as plt
plt.figure(figsize=(7,7))# figsize 그림그릴 판
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i],cmap = 'gray')
plt.show()

'''
이 코드는 Matplotlib 라이브러리를 사용하여 7x7 크기의 서브플롯(subplot)을 생성하고, MNIST 데이터셋의 첫 번째 이미지 샘플을 시각화하는 코드입니다.

첫째 줄에서는 그림을 그리기 위한 판의 크기를 정의하고, figsize 매개변수를 사용하여 (7,7) 크기로 설정합니다.

for 루프에서는 i가 0에서 48까지 반복하면서, 7x7 서브플롯 중 하나에 이미지를 시각화합니다.

plt.subplot() 함수는 그리드의 위치를 지정하는데 사용됩니다. 여기서는 7x7 그리드에서 i+1번째 위치를 지정합니다.

plt.axis() 함수를 사용하여 서브플롯의 x축과 y축 눈금선을 비활성화하고, plt.imshow() 함수를 사용하여 MNIST 데이터셋에서 첫 번째 샘플의 i번째 이미지를 그립니다. cmap 매개변수를 'gray'로 설정하여 흑백 이미지를 표시합니다.

마지막으로 plt.show() 함수를 사용하여 그림을 표시합니다.
'''