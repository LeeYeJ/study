#현 파일 3번문제 (말과 사람 혹은 가위바위보)
#4번 자연어처리
#5번 시계열
##https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

#넘파이까지 저장

path = 'd:/study_data/_data/Men/'
save_path = 'd:/study_data/_save/Men/'

# np.save(save_path+'파일명',arr=??)

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten
from sklearn.model_selection import train_test_split
import time

start_time = time.time()

#이미지 전처리
xy_datagen = ImageDataGenerator(
    rescale=1./255,         #'.':부동소수점으로 연산해라  #MinMaxscaler:정규화(Nomalization)=이미지: /255
) #이러한 옵션(데이터 증폭옵션)들 중 선택해서 이미지 전처리 가능 (훈련데이터) 


#D드라이브에서 데이터 가져오기 #변환
datasets = xy_datagen.flow_from_directory( #이미지제너레이터는 폴더별로 라벨값 부여
    'd:/study_data/_data/Men/', #분류된 폴더의 상위폴더까지 지정  #directory=폴더
    target_size=(150,150),        #수집한 데이터마다 이미지 사진크기 다르므로 이미지크기 동일하게 고정
    batch_size=3309,         # 전체 데이터를 쓰려면 160(전체 데이터 갯수) 이상을 넣어라 
    class_mode='binary', #0,1로 구별(nomal,ad) / 0,1,2(가위,바위,보)// #원핫사용한 경우 => 'categorical'
    color_mode='rgb',
    # color_mode='rgb', # 컬러부분이 3이됨
    shuffle=True,
)

x_datasets = datasets[0][0]
y_datasets = datasets[0][1]

print(datasets[0][0].shape)
print(datasets[0][1].shape)

print(x_datasets.shape) #(500, 400, 400, 3)
print(y_datasets.shape) #(500,)

x_train,x_test,y_train,y_test = train_test_split(
    x_datasets,y_datasets, shuffle=True, train_size=0.7,
)

print(x_train.shape)


np.save(save_path+'keras56_7_x_train.npy',arr =x_train) #xy_train[0][0]이 저 경로에 저장됨
np.save(save_path+'keras56_7_x_test.npy',arr =x_test)
np.save(save_path+'keras56_7_y_train.npy',arr =y_train)
np.save(save_path+'keras56_7_y_test.npy',arr =y_test)


end_time = time.time()

print(np.round(end_time-start_time,2))


# print(x_train.shape)
# print(x_test.shape)
# print(y_train)
# print(y_test)