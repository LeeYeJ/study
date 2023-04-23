from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Flatten,Conv2D,LSTM
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

#ReduceLROnPlateau 써보기

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kcal/'
path_save = './_save/kcal/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항
print(train_csv.shape, test_csv.shape)
# (7500, 10) (7500, 9)

# 1.3 결측지
# print(train_csv.isnull().sum())
# print(train_csv.info())

# 1.4 라벨인코딩( object 에서 )
le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
# print(len(train_csv.columns))
# print('==============')
# print(train_csv.info())
# print('===================')
train_csv=train_csv.dropna()
print(train_csv.shape)


# 1.5 x, y 분리
x = train_csv.drop(['Calories_Burned','Height(Remainder_Inches)','Weight_Status'], axis=1)
y = train_csv['Calories_Burned']

test_csv = test_csv.drop(['Height(Remainder_Inches)','Weight_Status',], axis=1)
# x = train_csv.drop(['Calories_Burned'], axis=1)
# y = train_csv['Calories_Burned']

print(x.shape,y.shape) #(7500, 9) (7500,)

# 1.6 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=35468264, shuffle=True)

print(x_train.shape) #(6000, 9)
print(x_test.shape) #(1500, 9)
print(test_csv.shape) #(7500, 9)

# 1.7 Scaler
# scaler = MinMaxScaler() # 여기서 어레이 형태로 해서 아래 리쉐잎때 변환안해줘도됨
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# test_csv = scaler.transform(test_csv)

print(x_train)

# 2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=8))
# model.add(Dense(64))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(8))
# model.add(Dense(1))

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
x_train= x_train.reshape(6000,7,1)
x_test= x_test.reshape(1500,7,1)
test_csv = test_csv.to_numpy()
test_csv = test_csv.reshape(7500,7,1)# test파일도 모델에서 돌려주니까 리쉐잎 해줘야됨.


model = Sequential()
model.add(LSTM(32,input_shape=(7,1),activation='linear',return_sequences=True))
model.add(LSTM(16,activation='linear'))
model.add(Dense(32,activation='selu'))
model.add(Dense(42))
model.add(Dense(13))
model.add(Dropout(0.1))
model.add(Dense(24))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

# model = Sequential()
# model.add(Conv2D(7,(2,1),input_shape=(8,1,1)))
# model.add(Conv2D(8,(2,1),activation='relu'))
# model.add(Flatten())
# model.add(Dense(9,activation='relu'))
# model.add(Dense(6))
# model.add(Dense(1))

# input1 = Input(shape=(78,))
# dense1 = Dense(32)(input1)
# drop1 = Dropout(0.2)(dense1)
# dense2 = Dense(64, activation='relu')(drop1)
# dense3 = Dense(64)(dense2)
# dense4 = Dense(32,activation='relu')(dense3)
# dense5 = Dense(35)(dense4)
# drop2 = Dropout(0.2)(dense5)
# output1 = Dense(1)(drop2)
# model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=300, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=2000, batch_size=30, verbose=1, validation_split=0.2, callbacks=[es])

model.save('./_save/kcal/kcal_save_model01.h5')

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# RMSE 함수 정의
def RMSE(y_test,y_pre):
    return np.sqrt(mean_squared_error(y_test,y_pre)) #정의
rmse=RMSE(y_test,y_predict) #사용
print('RMSE :',rmse)

# 4.1 내보내기
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')

y_submit = model.predict(test_csv)
y_submit = pd.DataFrame(y_submit)
# y_submit = y_submit.fillna(y_submit.mean()) # mean -> nan값을 평균값으로 대체해준다 
y_submit = y_submit.fillna(y_submit.median()) # median -> nan값을 중간값으로 대체해준다
# y_submit = y_submit.fillna(y_submit.mode()[1]) # mode -> nan값을 최빈값으로 대체해준다                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
y_submit = np.array(y_submit)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Calories_Burned'] = y_submit
submission.to_csv(path_save + 'kcal_01' + date + '.csv')

# import matplotlib.pyplot as plt 
# import matplotlib
# plt.rcParams['font.family'] = 'Malgun Gothic' # 한글깨짐 해결 #다르 폰트 필요하면 윈도우 폰트 파일에 추가해줘야됨 # 상용할땐 나눔체로 쓰자.
# plt.figure(figsize=(100,100)) #그래프의 사이즈 , 단위는 inch
# plt.plot(hist.history['loss'],marker='.',c='red',label='loss') # 순서대로 갈때는 x명시할 필요 없을
# plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss') #marker='.' 점점으로 표시->선이 됨
# plt.title('집값')
# plt.xlabel('epochs')
# plt.ylabel('loss , val_loss')
# plt.legend() # 선에 이름 표시
# plt.grid() #격자 표시
# plt.show()
