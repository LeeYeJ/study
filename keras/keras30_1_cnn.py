from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten # cnn 하겠다는것

model = Sequential()
model.add(Conv2D(7, (2,2), input_shape=(8,8,1))) # 7장으로 늘렸다(연산량이 늘어남)/(2,2)로 자른다(자르는 크기)/input_shape=(5,5,1)이미지 형태 흑백=1, 컬러=3(5헹5열)
#출력(N,7,7,7) -> (batcj_size. rows, colums, channels)가 다음부턴 filters가 됨
model.add(Conv2D(filters=4, # 필터 (아웃풋 노드의 갯수)
                 kernel_size=(3,3), #커널 사이즈
                 activation='relu')
          ) # 출력 (N,5,5,4) ->(batcj_size. rows, colums, channels)
model.add(Conv2D(10,(2,2),)) # 출력 (N, 4, 4, 10)
# 중첩하는 이윤 이미지는 보통 가장자리보단 중앙에 이미지가 있기때문에 중첩하며 연산이 많아지니 더 좋다.
model.add(Flatten()) # 상단의 데이터를 펴는 작업 #출력 (N, 4*4*10) -> (N,160) 이제 2차원이 됐으니 Dense로 받자
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))


model.summary()



