# import numpy as np
# import pandas as pd
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score,mean_squared_error
# from tensorflow.python.keras.callbacks import EarlyStopping
# from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

# #데이터 불러오기
# path = './_data/AI/'
# path_save='./_save/AI/'
# train_csv= pd.read_csv(path +'train_data.csv')
# test_csv=pd.read_csv(path + 'test_data.csv')

# print(train_csv)
# print(train_csv.shape) # (2463, 8)

# print(test_csv)
# print(test_csv.shape) #(7389, 8)

# #데이터 결측치 제거
# print(train_csv.isnull().info()) # 결측값 없음

# #데이터 x,y분리
# x=train_csv.drop(['','casual','registered'], axis=1)
# print(x.columns)

# y=train_csv['count']
# print(y)

# # 데이터 split
# x_train,x_test,y_train,y_test=train_test_split(
#     x,y,shuffle=True,random_state=3625114,train_size=0.9
# )
# print(x_train.shape,x_test.shape) # (9797, 8) (1089, 8)
# print(y_train.shape,y_test.shape) # (9797,) (1089,)

# # scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# # scaler= StandardScaler()
# # scaler= RobustScaler()
# scaler= MaxAbsScaler()
# scaler.fit(x_train) # fit의 범위가 x_train이다 
# x_train=scaler.transform(x_train) #변환시키라
# x_test=scaler.transform(x_test)

# #test 파일도 스케일링 해줘야됨!!!!!!!!!
# test_csv=scaler.transform(test_csv)

# # 모델 구성
# model=Sequential()
# model.add(Dense(5,input_dim=8))
# model.add(Dense(6))
# model.add(Dense(6,activation='relu')) # 음수값 조정할때 활성화 함수나 한정화 함수로 각 층에서 던져줄때 조정해줄수있음 
# model.add(Dense(5,activation='relu')) # (예를들면 음수에서 양수화해서 던져줌)
# model.add(Dense(6,activation='relu'))
# model.add(Dense(6))
# model.add(Dense(6))
# model.add(Dense(1,activation='sigmoid'))

# from sklearn.metrics import f1_score, accuracy_score
# #컴파일 훈련
# model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
# es=EarlyStopping(mode='min',monitor='val_loss',patience=20,restore_best_weights=True)
# model.fit(x_train,y_train,epochs=500,batch_size=200,validation_split=0.1,callbacks=[es])

# #평가
# loss= model.evaluate(x_test,y_test)
# print('loss :',loss)

# y_pred = model.predict(x_test)

# #f1 스코어
# f1_score = f1_score(y_test, y_pred, average='macro')
# print('f1', f1_score)


# #카운트값 빼기
# y_submit = model.predict(test_csv)
# # print(y_submit)


# #카운트값 넣어주기
# submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)
# # print(submission)

# #제출파일의 count에 y_submit을 넣어준다.
# submission['count'] = y_submit
# # print(submission)

# submission.to_csv(path_save + 'submit_0308_1941.csv')




# lof모델

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load train and test data
path='./_data/AI/'
save_path= './_save/AI/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Combine train and test data
# data = pd.concat([train_data, test_data], axis=0).values # 넘파이 변환
# print(type(data))

train_data = train_data.values
test_data = test_data.values
print(type(train_data))

scaler = MinMaxScaler()  # scaler 객체 생성
scaler_data = scaler.fit_transform(train_data)
test_data=scaler.transform(test_data)

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

# # LSTM AE 모델 구축
# input_shape = train_data.shape[0:]
# print(input_shape) # (2463, 8)
# latent_dim = 64

# inputs = Input(shape=input_shape)
# print(inputs.shape) #(None, 2463, 8)

# # Encoder
# encoded = LSTM(latent_dim)(inputs)

# # Decoder
# decoded = RepeatVector(input_shape[0])(encoded)
# decoded = LSTM(input_shape[1], return_sequences=True)(decoded)

# # Autoencoder
# autoencoder = Model(inputs, decoded)
# autoencoder.compile(optimizer='adam', loss='mae')

# # 데이터 학습
# # x_train, x_test = train_test_split(train_data)
# print(train_data.shape)
# autoencoder.fit(train_data, epochs=10, batch_size=32)

# # LSTM AE를 통해 데이터의 특징 추출
# encoder = Model(inputs, encoded)
# encoded_data = encoder.predict(test_data)

# Preprocess data
# ...

# Train isolation forest model on train data
model = IsolationForest(random_state=324541454,
                        n_estimators=3000, max_samples=200, contamination=0.04, max_features=7)
'''
random_state: 난수 발생 시드값입니다. 이 값을 고정하면 모델이 항상 같은 결과를 출력합니다.
n_estimators: Isolation Tree(결정 트리의 집합)의 개수입니다. 이 값이 클수록 모델의 정확도는 높아지지만, 계산 시간이 늘어납니다.
max_samples: 각 Isolation Tree에서 사용할 샘플의 최대 개수입니다. 이 값을 작게 설정하면 이상치를 탐지하는 데 민감해집니다.
contamination: 전체 데이터셋 중 이상치로 판단할 데이터셋의 비율입니다. 이 값이 작을수록 이상치로 판단할 데이터셋이 적어지므로, 더 엄격한 기준으로 이상치를 탐지합니다.
max_features: 각 Isolation Tree에서 사용할 최대 특징의 수입니다. 이 값을 작게 설정하면 이상치를 탐지하는 데 민감해집니다.
'''
model.fit(train_data) # 트레인 데이터로 훈련

joblib.dump(model, './_save/AI_save_model/isolation_forest6.joblib') # 가중치 저장

# andom_state=640874, n_estimators=500, max_samples=1000, contamination=0.05, max_features=5)

# Predict anomalies in test data
predictions = model.predict(test_data)

# Save predictions to submission file
new_predictions = [0 if x == 1 else 1 for x in predictions]
submission['label'] = pd.DataFrame({'Prediction': new_predictions})

#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

submission.to_csv(save_path+'submit_air_'+date+ '.csv', index=False)

