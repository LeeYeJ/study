# stride 보폭 -> 커널 사이즈의 보폭

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten # cnn 하겠다는것

model = Sequential()
model.add(Conv2D(7, (2,2), 
                 padding='same',  # 패딩 적용되어서 사이즈가 변하지 않음(유지) (커널사이즈가 몇이든) # 대신 커널사이즈에 따른 파람 값은 달라진다.
                 strides=2, # 보폭 조절
                 input_shape=(9,9,1))) # 7장으로 늘렸다(연산량이 늘어남)/(2,2)로 자른다(자르는 크기)/input_shape=(5,5,1)이미지 형태 흑백=1, 컬러=3(5헹5열)
#출력(N,7,7,7) -> (batcj_size. rows, colums, channels)가 다음부턴 filters가 됨
model.add(Conv2D(filters=4, # 필터 (아웃풋 노드의 갯수)
                 kernel_size=(3,3), #커널 사이즈
                #  padding='valid' # 디폴트값
                #  padding='same',
                 activation='relu')
          ) # 출력 (N,5,5,4) ->(batcj_size. rows, colums, channels)
model.add(Conv2D(10,(2,2),)) # 출력 (N, 4, 4, 10)
model.add(Flatten()) # 상단의 데이터를 펴는 작업 #출력 (N, 4*4*10) -> (N,160) 이제 2차원이 됐으니 Dense로 받자 / 연산이 아니고 모양만 바꿔준다.
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()



