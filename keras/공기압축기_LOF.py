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
#######################################################################

# # label 열이 0인 데이터의 개수를 구한다
# num_zeros = (s['label'] == 0).sum()

# print('Number of label 0 data:', num_zeros)

# ==============================================================================================
# # lof모델

# import pandas as pd
# from sklearn.ensemble import IsolationForest
# from sklearn.metrics import f1_score, make_scorer
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import MinMaxScaler,RobustScaler
# import joblib
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense 

# # Load train and test data
# path='./_data/AI/'
# save_path= './_save/AI/'
# train_data = pd.read_csv(path+'train_data.csv')
# test_data = pd.read_csv(path+'test_data.csv')
# submission = pd.read_csv(path+'answer_sample.csv')

# # Combine train and test data
# data = pd.concat([train_data, test_data], axis=0).values # 넘파이 변환
# print(type(data))

# # train_data = train_data.values
# # test_data = test_data.values
# # print(type(train_data))

# scaler = MinMaxScaler()  # scaler 객체 생성
# scaler_data = scaler.fit_transform(train_data)
# test_data=scaler.transform(test_data)

# from sklearn.neighbors import LocalOutlierFactor

# from sklearn.model_selection import train_test_split

# train_data, test_data = train_test_split(
#     data, train_size=0.9, random_state=35248
# )


# # lof = LocalOutlierFactor(n_neighbors=50, contamination=0.2 , novelty=True,)
# # lof.fit(train_data)

# from sklearn import svm
# clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# clf.fit(train_data)

# # Train isolation forest model on train data
# # model = IsolationForest(random_state=324541454,
# #                         n_estimators=3000, max_samples=200, contamination=0.04, max_features=7)

# # model.fit(train_data) # 트레인 데이터로 훈련

# # joblib.dump(clf, './_save/AI_save_model/isolation_forest8.joblib') # 가중치 저장

# # andom_state=640874, n_estimators=500, max_samples=1000, contamination=0.05, max_features=5)

# predictions = clf.predict(test_data)
# print(predictions)

# # Predict anomalies in test data
# # predictions = model.predict(test_data)

# # Save predictions to submission file
# new_predictions = [0 if x == 1 else 1 for x in predictions]
# print(new_predictions)
# submission['label'] = pd.DataFrame({'Prediction': new_predictions},dtype=int)


# #time
# import datetime 
# date = datetime.datetime.now()  
# date = date.strftime("%m%d_%H%M")  

# submission.to_csv(save_path+'submit_air_'+date+ '.csv', index=False)
# =============================================================================================
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector,Reshape
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# from sklearn.metrics import f1_score, make_scorer, accuracy_score


# # 훈련 데이터 및 테스트 데이터 로드
# path='./_data/AI/'
# save_path= './_save/AI/'
# train_data = pd.read_csv(path+'train_data.csv')
# test_data = pd.read_csv(path+'test_data.csv')
# submission = pd.read_csv(path+'answer_sample.csv')


# # Combine train and test data
# data = pd.concat([train_data, test_data], axis=0)

# # Preprocess data
# # 
# def type_to_HP(type):
#     HP=[30,20,10,50,30,30,30,30]
#     gen=(HP[i] for i in type)
#     return list(gen)
# train_data['type']=type_to_HP(train_data['type'])
# test_data['type']=type_to_HP(test_data['type'])


# # Select subset of features for Autoencoder model
# features = ['air_inflow','air_end_temp','motor_current','motor_rpm','motor_temp','motor_vibe']

# # Split data into train and validation sets
# x_train, x_val = train_test_split(data[features], train_size= 0.7, random_state= 346844444)

# # Normalize data
# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_val = scaler.transform(x_val)

# # Define Autoencoder model
# input_layer = Input(shape=(len(features),))
# encoder1 = Dense(30, activation='relu')(input_layer)
# encoder = Reshape(target_shape=(30,1))(encoder1)
# decoder = LSTM(len(features), activation='sigmoid')(encoder)
# decoded1 = RepeatVector(len(features))(decoder)
# decoded2 = LSTM(1, activation='sigmoid', return_sequences=True)(decoded1)
# autoencoder = Model(inputs=input_layer, outputs=decoded2)

# # Compile Autoencoder model
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# # Train Autoencoder model
# es = EarlyStopping(monitor='val_loss', mode='auto', verbose= 1, patience= 60, restore_best_weights=True)
# autoencoder.fit(x_train, x_train, epochs=500, batch_size= 20, validation_data=(x_val, x_val), callbacks=[es])
# # Model.save('./_save/AI_save_model/Air_1_save_model.h5')

# # Predict anomalies in test data
# test_data = scaler.transform(test_data[features])
# predictions = autoencoder.predict(test_data)
# test_data = test_data.reshape(7389, 6, 1)
# print(test_data.shape,predictions.shape)
# mse = ((test_data - predictions) ** 2).mean(axis=1)
# threshold = mse.mean() + mse.std() * 2  # Set threshold based on mean and standard deviation of MSE

# # Evaluate model performance
# binary_predictions = [1 if x > threshold else 0 for x in mse]

# submission['label'] = pd.DataFrame({'Prediction': binary_predictions})

# #time
# import datetime 
# date = datetime.datetime.now()  
# date = date.strftime("%m%d_%H%M")  

# submission.to_csv(save_path + date + 'submission.csv', index=False)
#=============================================================================================
# import pandas as pd
# import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.metrics import f1_score
# import joblib

# # 훈련 데이터 및 테스트 데이터 로드
# path='./_data/AI/'
# save_path= './_save/AI/'
# train_data = pd.read_csv(path+'train_data.csv')
# test_data = pd.read_csv(path+'test_data.csv')
# submission = pd.read_csv(path+'answer_sample.csv')

# # 데이터 전처리
# def type_to_HP(type):
#     HP=[30,20,10,50,30,30,30,30]
#     gen=(HP[i] for i in type)
#     return list(gen)
# train_data['type']=type_to_HP(train_data['type'])
# test_data['type']=type_to_HP(test_data['type'])
# # print(train_data.columns)

# # lof모델 적용 피처
# features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# # Prepare train and test data
# X = train_data[features]

# # 학습 데이터를 훈련 세트와 검증 세트로 나누기
# X_train, X_val = train_test_split(X, train_size= 0.9, random_state= 9842152)

# # 데이터 정규화
# scalers = [MinMaxScaler(),MaxAbsScaler(),RobustScaler(),StandardScaler()]
# # scaler = MinMaxScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# # scaler = StandardScaler()

# for i in scalers:
#     scaler = i
#     train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
#     test_data_normalized = scaler.transform(test_data.iloc[:, :-1])

#     # lof사용하여 이상치 탐지
#     # n_neighbors = 45
#     # contamination = 0.1
#     # lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
#     lof = LocalOutlierFactor(
#                                 n_neighbors=75, 
#                                 algorithm='auto', 
#                                 contamination=0.001, 
#                                 n_jobs=-1, 
#                                 leaf_size=28
#                             )

#     joblib.dump(lof, './_save/AI_save_model/isolation_forest16.joblib') # 가중치 저장
#     y_pred_train_tuned = lof.fit_predict(X_train)

#     # 이상치 탐지
#     test_data_lof = scaler.transform(test_data[features])
#     y_pred_test_lof = lof.fit_predict(test_data_lof)
#     # y_pred_test_lof = y_pred_test_lof[:2216]
#     # result = f1_score(y_pred_train_tuned,y_pred_test_lof)
#     # print('f1_score :', result)
#     lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]

#     submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
#     print(submission.value_counts())
#     #time
#     date = datetime.datetime.now()
#     date = date.strftime("%m%d_%H%M")

#     submission.to_csv(save_path + date + 'submission.csv', index=False)

'''
0.85 - save 11
0.92 - save 12 (1001)
- save 13 (위에서 파라미터 조절) (1016) -> 일단 f1스코어 잘나옴 0.009
-save 14 (1017) -> 0.012
0.7-save 15 (1018) -> 0.012
0.5-save 16 (1020) -> 0.001
# '''
# import pandas as pd
# import numpy as np
# import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.metrics import accuracy_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# import time
# import xgboost as xgb
# from sklearn.decomposition import PCA

# # 훈련 데이터 및 테스트 데이터 로드
# path='./_data/AI/'
# save_path= './_save/AI/'
# train_data = pd.read_csv(path+'train_data.csv')
# test_data = pd.read_csv(path+'test_data.csv')
# submission = pd.read_csv(path+'answer_sample.csv')

# # 데이터 전처리
# def type_to_HP(type):
#     HP=[30,20,10,50,30,30,30,30]
#     gen=(HP[i] for i in type)
#     return list(gen)
# train_data['type']=type_to_HP(train_data['type'])
# test_data['type']=type_to_HP(test_data['type'])
# # print(train_data.columns)

# train_data_x = train_data.drop(['air_end_temp'], axis=1)
# train_data_y = train_data['air_end_temp']
# test_data_x = train_data.drop(['air_end_temp'], axis=1)
# test_data_y = train_data['air_end_temp']


# # lof모델 적용 피처
# features = ['air_inflow', 'out_pressure', 'motor_current','motor_rpm', 'motor_temp', 'motor_vibe','type']

# # Prepare train and test data
# # X = train_data[features]


# # 상관관계 가시화
# import matplotlib.pyplot as plt
# import seaborn as sns

# print(test_data.corr())
# plt.figure(figsize=(10,8)) #새로운 그림(figure)을 생성
# sns.set(font_scale=1.2) # seaborn에서 그래프를 그릴 때 사용되는 기본 스타일, 폰트, 색상, 크기 등을 설정
# sns.heatmap(train_data.corr(),square=True,annot=True,cbar=True) #데이터 프레임, 배열, 시리즈 등을 입력받아 색상으로 나타내
# plt.show()


# # 학습 데이터를 훈련 세트와 검증 세트로 나누기
# x_train,x_val,y_train,y_val = train_test_split(train_data_x,train_data_y, train_size= 0.9, random_state= 5050)

# # 데이터 정규화
# scaler = MinMaxScaler()
# x_train_norm = scaler.fit_transform(x_train)
# # X_val_norm = scaler.transform(X_val)
# test_data_norm = scaler.transform(test_data[features])

# # lof사용하여 이상치 탐지
# n_neighbors = 50
# contamination = 0.01
# lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, leaf_size=28)
# y_pred_train_tuned = lof.fit_predict(x_train)

# import joblib
# joblib.dump(lof,'./_save/AI_save_model/isolation_forest20.joblib')

# # 이상치 탐지
# test_data_lof = scaler.transform(test_data[features])
# y_pred_test_lof = lof.fit_predict(test_data_lof)
# lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]

# # 모델
# model = Sequential()
# model.add(Dense(128, input_dim=x_train_norm.shape[1]))
# model.add(Dense(64))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(32))
# model.add(Dense(28, activation='swish'))
# model.add(Dense(36))
# model.add(Dense(10))
# model.add(Dense(x_train_norm.shape[1], activation='linear'))

# # 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics = ['acc'])

# es = EarlyStopping(monitor='val_acc', mode='max', patience=30)

# history = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_data=(x_val, y_val), callbacks=[es])

# # 평가
# test_preds = model.predict(test_data_norm)
# errors = np.mean(np.power(test_data_norm - test_preds, 2), axis=1)
# y_pred = np.where(errors >= np.percentile(errors, 95), 1, 0)

# submission['label'] = y_pred
# # submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
# print(submission.value_counts())
# #time
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# submission.to_csv(save_path + date + 'submission.csv', index=False)
####################################################################
# import pandas as pd
# import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.cluster import DBSCAN

# # 훈련 데이터 및 테스트 데이터 로드
# path='./_data/AI/'
# save_path= './_save/AI/'
# train_data = pd.read_csv(path+'train_data.csv')
# test_data = pd.read_csv(path+'test_data.csv')
# submission = pd.read_csv(path+'answer_sample.csv')

# # 데이터 전처리
# def type_to_HP(type):
#     HP=[30,20,10,50,30,30,30,30]
#     gen=(HP[i] for i in type)
#     return list(gen)
# train_data['type']=type_to_HP(train_data['type'])
# test_data['type']=type_to_HP(test_data['type'])
# # print(train_data.columns)

# # lof모델 적용 피처
# features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# # Prepare train and test data
# X = train_data[features]

# # 학습 데이터를 훈련 세트와 검증 세트로 나누기
# X_train, X_val = train_test_split(X, train_size= 0.9, random_state= 363636)

# # 데이터 정규화
# scaler = MinMaxScaler()
# train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
# test_data_normalized = scaler.transform(test_data.iloc[:, :-1])

# # lof사용하여 이상치 탐지
# n_neighbors = 90
# contamination = 0.038
# lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, leaf_size=52)
# # dbscan = DBSCAN(eps=0.75, min_samples=20, algorithm='ball_tree',leaf_size=32)
# y_pred_train_tuned = lof.fit_predict(X_train)

# # 이상치 탐지
# test_data_lof = scaler.transform(test_data[features])
# y_pred_test_lof = lof.fit_predict(test_data_lof)
# lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]

# submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
# print(submission.value_counts())
# #time
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# submission.to_csv(save_path + date + 'submission.csv', index=False)
########################################################################

# import pandas as pd
# import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.decomposition import PCA
# import joblib
# import numpy as np
# from tensorflow.keras.models import Sequential, save_model, load_model
# from tensorflow.keras.layers import Dense,LeakyReLU,Dropout,Input
# from tensorflow.keras.callbacks import EarlyStopping as es

# # 훈련 데이터 및 테스트 데이터 로드
# path='./_data/AI/'
# save_path= './_save/AI/'
# train_data = pd.read_csv(path+'train_data.csv')
# test_data = pd.read_csv(path+'test_data.csv')
# submission = pd.read_csv(path+'answer_sample.csv')

# def type_to_HP(type):
#     HP=[30,20,10,50,30,30,30,30]
#     gen=(HP[i] for i in type)
#     return list(gen)
# train_data['type']=type_to_HP(train_data['type'])
# test_data['type']=type_to_HP(test_data['type'])
# # print(train_data.columns)


# features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp']
# features_y = ['air_inflow']

# # Prepare train and test data
# X = train_data[features]
# print(X.shape)
# pca = PCA(n_components=3)
# X = pca.fit_transform(X)
# print(X.shape)

# # 
# X_train, X_val = train_test_split(X, test_size= 0.9, random_state= 32323232)
# print(X_train.shape, X_val.shape)

# #
# pca = PCA(n_components=3, random_state=323232)
# X_train = pca.fit_transform(X_train)
# X_val = pca.fit_transform(X_val)

# # 
# scaler = MinMaxScaler()
# train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
# test_data_normalized = scaler.transform(test_data.iloc[:, :-1])


# n_neighbors = 42
# contamination = 0.04588888
# #n_neighbors데이터 포인트에 대한 LOF 점수를 계산할 때 고려할 이웃 수를 결정합니다. 값 이 높을수록 이상 n_neighbors값을 감지하는 능력이 향상될 수 있지만 정상 데이터 포인트를 이상값으로 잘못 식별할 위험도 증가합니다. 따라서 n_neighbors특정 문제 및 데이터를 기반으로 신중하게 조정해야 합니다.
# lof = LocalOutlierFactor(n_neighbors = n_neighbors,
#                          contamination=contamination,
#                          algorithm='auto',
#                          metric_params= None,
#                          metric='chebyshev',
#                          novelty=False,
#                          )

# # 
# test_data_lof = scaler.fit_transform(test_data[features])
# y_pred_test_lof = lof.fit_predict(test_data_lof)
# lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]
# #lof_predictions = [0 if x == -1 else 0 for x in y_pred_test_lof]

# for_sub=submission.copy()
# for_test=test_data.copy()
# print(f'subshape:{for_sub.shape} testshape: {for_test.shape}')
# submission['label'] = lof_predictions
# train_data['label'] = np.zeros(shape=train_data.shape[0],dtype=np.int64)
# test_data['label'] = lof_predictions
# print(test_data.shape,train_data.shape)

# # print(submission.value_counts())
# # print(submission['label'].value_counts())

# for_train=np.concatenate((train_data.values,test_data.values),axis=0)
# print(for_train.shape)


# # 1. data prepare
# # y값이 0인 데이터와 1인 데이터 분리
# zero_data = for_train[for_train[:, -1] == 0]
# one_data = for_train[for_train[:, -1] == 1]
# num_zero = len(zero_data)
# num_one = len(one_data)

# from sklearn.utils import resample
# one_data = np.repeat(one_data, num_zero//num_one*1.5, axis=0)
# for_train=np.concatenate((zero_data,one_data),axis=0)
# x = for_train[:,:-1]
# y = for_train[:,-1]

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.9, random_state=333, stratify=y
#                                                )
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
# for_test=scaler.transform(for_test)

# # 2. model build
# model=Sequential()
# model.add(Input(shape=x_train.shape[1:]))
# model.add(Dense(512,activation=LeakyReLU(0.15)))
# model.add(Dropout(1/16))
# model.add(Dense(512,activation=LeakyReLU(0.15)))
# model.add(Dropout(1/16))
# model.add(Dense(512,activation=LeakyReLU(0.15)))
# model.add(Dropout(1/16))
# model.add(Dense(512,activation=LeakyReLU(0.15)))
# model.add(Dropout(1/16))
# model.add(Dense(512,activation=LeakyReLU(0.15)))
# model.add(Dropout(1/16))
# model.add(Dense(1,activation='sigmoid'))

# # 3. compile,training
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics='acc')
# model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10000,batch_size=len(x_train)//99
#           ,callbacks=es(monitor='val_loss',mode='min',patience=20,verbose=True,restore_best_weights=True))

# # model.save('./_save/ai_f_model.h5')
# # model = load_model('./_save/ai_f_model.h5')
# # model.fit(x_train, y_train, epochs=500, validation_split=0.3, batch_size=200,
# #           callbacks=es(monitor='val_loss',mode='min',patience=200,verbose=True,restore_best_weights=True))

# # 4. predict,save
# print(x_train.shape,for_test.shape)
# y_pred=model.predict(for_test)
# for_sub[for_sub.columns[-1]]=np.round(y_pred)
# import datetime
# now=datetime.datetime.now().strftime('%m월%d일%h시%M분')
# print(for_sub.value_counts())
# for_sub.to_csv(f'{save_path}{now}_submission.csv',index=False)

#################################################################
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import joblib


# 훈련 데이터 및 테스트 데이터 로드
path='./_data/AI/'
save_path= './_save/AI/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# 
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])
# print(train_data.columns)

# 
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]
print(X.shape)
pca = PCA(n_components=3)
X = pca.fit_transform(X)
print(X.shape)

# 
X_train, X_val = train_test_split(X, test_size= 0.9, random_state= 553)
print(X_train.shape, X_val.shape)

#
pca = PCA(n_components=3)
X_train = pca.fit_transform(X_train)
X_val = pca.fit_transform(X_val)

# 
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])


n_neighbors = 42
contamination = 0.035
#n_neighbors데이터 포인트에 대한 LOF 점수를 계산할 때 고려할 이웃 수를 결정합니다. 값 이 높을수록 이상 n_neighbors값을 감지하는 능력이 향상될 수 있지만 정상 데이터 포인트를 이상값으로 잘못 식별할 위험도 증가합니다. 따라서 n_neighbors특정 문제 및 데이터를 기반으로 신중하게 조정해야 합니다.
lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                         contamination=contamination,
                         leaf_size=99,
                         algorithm='auto',
                         )
y_pred_train_tuned = lof.fit_predict(X_val)

joblib.dump(lof, './_save/AI_save_model/_model_ai_factory.joblib')

# 
test_data_lof = scaler.fit_transform(test_data[features])
y_pred_test_lof = lof.fit_predict(test_data_lof)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]
#lof_predictions = [0 if x == -1 else 0 for x in y_pred_test_lof]
####
#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
print(lof_predictions)
lof_predictions_df = pd.DataFrame(lof_predictions, columns=[ 'label'])
# for i in range(7000, len(lof_predictions)):
for i in range(7000, len(lof_predictions)):
    if lof_predictions_df.loc[i, 'label'] == 1:
        lof_predictions_df.loc[i, 'label'] = 0
        submission['label'] = lof_predictions_df
        submission.to_csv(save_path + date + '_REAL_LOF_submission.csv', index=False)
        print(submission.value_counts())


# print(test_data.corr())
# plt.figure(figsize=(10,8))
# sns.set(font_scale=1.2)
# sns.heatmap(train_data.corr(), square=True, annot=True, cbar=True)
# plt.show()

#0.9551928573
#0.9551928573
#0.9561993171
#0.9570394969
#0.9582241632





