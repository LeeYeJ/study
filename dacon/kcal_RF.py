from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Flatten,Conv2D,LSTM
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd

parameters = [
    {'n_estimators':[100,200]},
    {'max_depth':[6,8,10,12]},
    {'min_sample_leaf':[3,4,7,10]},
    {'min_samples_split':[2,3,5,10]},
    {'n_jobs':[-1,2,4]}
]

# 파라미터 조합으로 2개 이상 엮을것

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12],'min_samples_leaf':[3,10]},
    {'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10],'min_samples_split':[2,3,5,10]},
    {'min_samples_split':[2,3,5,10],'max_depth':[6,8,10,12]}
]

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

# print(x_train)

# 2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=8))
# model.add(Dense(64))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(8))
# model.add(Dense(1))

# x_train= x_train.reshape(6000,9,1)
# x_test= x_test.reshape(1500,9,1)
# test_csv = test_csv.reshape(7500,9,1) # test파일도 모델에서 돌려주니까 리쉐잎 해줘야됨.


# n_splits = 5 # 디폴트값 5
# kfold = KFold(n_splits = n_splits, shuffle=True,random_state=123245) 

model = RandomizedSearchCV(RandomForestRegressor(),
                     parameters, 
                    #  cv = 5,  # 분류의 디폴트는 StratifiedKFold이다.
                    #  cv = kf,  
                     verbose=1, 
                     refit=True, # 최적의 값을 보관함 / 최적의 값을 출력 -> 통상적으로 True로 함
                    #  refit=False, # 모델이 돌아갈때 최적값을 저장하지 않음 -> False하면 최종 파라미터로 출력
                     n_jobs=-1)

# model = Sequential()
# model.add(LSTM(16,input_shape=(9,1),activation='linear',return_sequences=True))
# model.add(LSTM(16,input_shape=(9,1),activation='relu'))
# model.add(Dense(32,activation='selu'))
# model.add(Dense(42))
# model.add(Dense(13))
# model.add(Dropout(0.1))
# model.add(Dense(24))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(1))

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
# model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=300, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train)

# model.save('./_save/kcal/kcal_save_model01.h5')

#<trian>
print('최적의 매개변수 :',model.best_estimator_) # 가장 좋은 평가 뽑기
# 최적의 매개변수 : SVC(C=1, kernel='linear

print('최적의 파라미터 :',model.best_params_) # 가장 좋은 파람 뽑기
# 최적의 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'}

print('best_score_ :',model.best_score_) # 가장 좋은 점수
# best_score_ : 0.9916666666666668

#<test> 
print('model.score :',model.score(x_test,y_test)) # 테스트한 모델 스코어 (중요)
# model.score : 1.0

y_predict = model.predict(x_test)
print('r2_score :',r2_score(y_test,y_predict))
# acc : 1.0

y_pred_best = model.best_estimator_.predict(x_test)

# RMSE 함수 정의
def RMSE(y_test,y_pre):
    return np.sqrt(mean_squared_error(y_test,y_pre)) #정의
rmse=RMSE(y_test,y_pred_best)#사용
print('최적의 튠 RMSE :',rmse)

# y_pred_best = model.best_estimator_.predict(x_test)
# print('최적 튠 RMSE :',rmse)
# 최적 튠 ACC : 1.0 

# predict / best_estimator_ 값이 같음 -> 최적값 저장됐으니까

# # 4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)

# y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)
# print('r2 : ', r2)

# # RMSE 함수 정의
# def RMSE(y_test,y_pre):
#     return np.sqrt(mean_squared_error(y_test,y_pre)) #정의
# rmse=RMSE(y_test,y_predict) #사용
# print('RMSE :',rmse)

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
submission.to_csv(path_save + 'kcal_' + date + '.csv')

