#반전시키면 알아볼수있음? / 바이너리 -> 수치화/ numpy -> unique = padas -> value_counts

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#이미지 전처리
train_datagen = ImageDataGenerator(
    rescale=1./255,         #'.':부동소수점으로 연산해라  #MinMaxscaler:정규화(Nomalization)=이미지: /255
    horizontal_flip=True,   #수평을 반전(상하반전)  #주의) 숫자 6,9있을 경우 사용하면 안됨
    vertical_flip=True,     #좌우반전
    width_shift_range=0.1,  #10%만큼 좌우로 이동 할 수 있다
    height_shift_range=0.1, #데이터 증폭 : 10% 상하이동
    rotation_range=5,       #지정된 각도 범위내에서 임의로 원본이미지를 회전
    zoom_range=1.2,         #약 20%까지 확대
    shear_range=0.7,        #찌그러트림(밀림 강도)
    fill_mode='nearest',    #10% 이동한 이후 생기는 빈자리를 근접값(비슷한 수치)로 채움
) #이러한 옵션(데이터 증폭옵션)들 중 선택해서 이미지 전처리 가능 (훈련데이터) 

test_datagen = ImageDataGenerator(
    rescale=1./255,
) #평가데이터를 증폭한다는 것은 값을 조작한다는 의미=> 따라서, 통상적으로 testdata는 전처리x(증폭x) 


#D드라이브에서 데이터 가져오기 
xy_train = train_datagen.flow_from_directory( #이미지제너레이터는 폴더별로 라벨값 부여
    'd:/study_data/_data/brain/train/', #분류된 폴더의 상위폴더까지 지정  #directory=폴더
    target_size=(200,200),        #수집한 데이터마다 이미지 사진크기 다르므로 이미지크기 동일하게 고정
    batch_size=10,         # 전체 데이터를 쓰려면 160(전체 데이터 갯수) 이상을 넣어라 
    class_mode='binary', #0,1로 구별(nomal,ad) / 0,1,2(가위,바위,보)// #원핫사용한 경우 => 'categorical'
    color_mode='grayscale',
    # color_mode='rgb', # 컬러부분이 3이됨
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/', #분류된 폴더의 상위폴더까지 지정
    target_size=(200,200),        #수집한 데이터마다 이미지 사진크기 다르므로 이미지크기 동일하게 고정
    batch_size=10,    
    class_mode='binary', #0,1로 구별(nomal,ad) / 0,1,2(가위,바위,보)// #원핫사용한 경우 => 'categorical'
    color_mode='grayscale',
    # color_mode='rgb',
    shuffle=True,
)

'''
#수치화 된 것 확인가능(실행)
Found 160 images belonging to 2 classes. #xy_train #x_train.shape = (160,200,200,1)            #y_train.shape =(160,)
Found 120 images belonging to 2 classes. #xy_test  #x_test.shape = (120,200,200,1)로 바뀜(흑백) #y_test.shape  =(120,) : 이미지제너레이터는 폴더별로 라벨값 부여하므로 y는 (0,1)->120
# 카데고리별 개수 확인
# pandas - value_counts / numpy - unique 
'''

print(xy_train) 
# <keras.preprocessing.image.DirectoryIterator object at 0x0000028A4F535F70>
print(xy_train[0])
'''
(x[0] : array([[[[0.22352943],
         [0.22352943],
         [0.22352943],
         ...,
         [[0.        ],
         [0.        ],
         [0.        ],
         ...,
         [0.        ],
         [0.        ],
         [0.        ]]]]
y[0] : array([0., 1., 1., 1., 1.]
'''

# print(xy_train.shape) #error => #numpy, pandas만
print(len(xy_train))        # 32 /(160/5(batch_size))/ [0]~[31]까지 있음/ [0][0]=x, [0][1]=y
print(len(xy_train[0]))     # 2  (x,y)/ 첫번째 batch
print(xy_train[0][0])       # x 5개 들어가있음 (batch=5일때)
print(xy_train[0][1])       # y [0. 1. 1. 1. 0.]
print(xy_train[0][0].shape) #(5, 200, 200, 1)  #numpy형태라 shape가능
print(xy_train[0][1].shape) #(5,)

'''
x와 y가 합쳐진 iterator형태는 같이 넣어도 됨
batch_size=10일 때  
print(xy_train[0][0])        #x 10개 들어가있음
print(xy_train[0][1])        #[0. 1. 0. 1. 1. 0. 1. 1. 1. 0.]
print(xy_train[0][0].shape)  #(10, 200, 200, 1)
print(xy_train[0][1].shape)  #(10,)
'''

print("===========================================================")
print(type(xy_train))  #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(200,200,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(15))
model.add(Dense(20,activation='relu'))
model.add(Dense(15))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics=['acc'])

# model.fit(xy_train[0][0],xy_train[0][1])  # 얘는 통으로 들어감
hist = model.fit_generator(xy_train,
                    epochs=100, # fit generator는 배치 사이즈까지 처리해서 한번에 훈련
                    steps_per_epoch=10,  # 전체 데이터에서 배치 나눈 값으로 주면 됨
                    validation_data=xy_test,
                    validation_steps=24) # 전체 데이터 나누기 배치

from sklearn.metrics import accuracy_score

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss :',loss[-1])

import matplotlib.pyplot as plt 
import matplotlib

# plt.subplot(1, 2, 1)
# plt.rcParams['font.family'] = 'Malgun Gothic' # 한글깨짐 해결 #다르 폰트 필요하면 윈도우 폰트 파일에 추가해줘야됨 # 상용할땐 나눔체로 쓰자.
# # plt.figure(figsize=(9,6)) #그래프의 사이즈 , 단위는 inch
# plt.plot(hist.history['loss'],marker='.',c='red',label='loss') # 순서대로 갈때는 x명시할 필요 없을
# plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss') #marker='.' 점점으로 표시->선이 됨
# plt.title('Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss,val_loss')
# plt.legend() # 범례 표시
# plt.grid() #격자 표시


# plt.subplot(1, 2, 2)
# plt.rcParams['font.family'] = 'Malgun Gothic' # 한글깨짐 해결 #다르 폰트 필요하면 윈도우 폰트 파일에 추가해줘야됨 # 상용할땐 나눔체로 쓰자.
# # plt.figure(figsize=(9,6)) #그래프의 사이즈 , 단위는 inch
# plt.plot(hist.history['loss'],marker='.',color='violet',label='loss') # 순서대로 갈때는 x명시할 필요 없을
# plt.plot(hist.history['val_loss'],marker='.',color='limegreen',label='val_loss') #marker='.' 점점으로 표시->선이 됨
# plt.title('ACC')
# plt.xlabel('epochs')
# plt.ylabel('acc,val_acc')
# plt.legend() # 범례 표시
# plt.grid() #격자 표시

# plt.tight_layout()
# plt.show()