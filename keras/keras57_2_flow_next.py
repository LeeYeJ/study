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

augment_size = 100 #argument_size증폭 / 

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
).next() # 한번 실행시키겠다.


##############.next() 사용 ################ 현 파일에선 통배치로 들어가서 .next로 첫 튜플에 있는 x,y를 가져온다.

print(x_data) #print(x_data[0]) # x y가 모두 포함
print(type(x_data)) #<class 'tuple'>
print(x_data[0]) # x데이터
print(x_data[1]) # y데이터
print(x_data[0].shape, x_data[1].shape) # (100, 28, 28, 1) (100,)
print(type(x_data)) #<class 'tuple'>
 
##############.next() 미사용 ################

# print(x_data[0]) # x y가 모두 포함
# print(x_data[0][0].shape) #(100, 28, 28, 1)
# print(x_data[0][1].shape) #(100,)

augment_size = 100 # 변환 데이터는 100개 중 49개

'''
iterator method

1) hasNext(): 다음 요소에 읽어 올 요소가 있는지 확인 하는 메소드 있으면 true, 없으면 false 를 반환한다. 

2) next(): 다음 요소를 가져온다. 

3) remove(): next()로 읽어온 요소를 삭제한다.
'''


import matplotlib.pylab as plt
plt.figure(figsize=(7,7))# figsize 그림그릴 판
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    # plt.imshow(x_data[0][0][i],cmap = 'gray') # x_data[0] 배치크기 포함 / .next()미사용
    plt.imshow(x_data[0][i], cmap='gray') # .next() 사용
plt.show()

