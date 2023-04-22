import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler,RobustScaler
import joblib


# Load train and test data
path='./_data/AI/'
save_path= './_save/AI/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Combine train and test data
data = pd.concat([train_data, test_data], axis=0).values # 넘파이 변환
print(type(data))

# train_data = train_data.values
# test_data = test_data.values
# print(type(train_data))

scaler = MinMaxScaler()  # scaler 객체 생성
scaler_data = scaler.fit_transform(data)
# test_data=scaler.transform(test_data)

# Train isolation forest model on train data
# model = IsolationForest(random_state=324541454,
#                         n_estimators=3000, max_samples=200, contamination=0.04, max_features=7)

# model.fit(train_data) # 트레인 데이터로 훈련

print(data.shape) #(9852, 8)

data = data.reshape(9852,8,1)
# print(data.shape) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

#2.모델
model= Sequential()
model.add(LSTM(16,input_shape=(3,1))) # 행 빼고 나머지/ 다른 모델들도 마찬가지임 / 32는 아웃풋 노드의 갯수
model.add(Dense(16,activation='relu'))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(10,activation='relu'))
model.add(Dense(16))
model.add(Dense(1 ,activation='sigmoid'))

model.compile(loss='binary', optimizer = 'adam')
model.fit(data,)

joblib.dump(model, './_save/AI_save_model/isolation_forest7.joblib') # 가중치 저장

# andom_state=640874, n_estimators=500, max_samples=1000, contamination=0.05, max_features=5)

predictions = model.predict(test_data)
print(predictions)

# Predict anomalies in test data
# predictions = model.predict(test_data)

# Save predictions to submission file
new_predictions = [0 if x == 1 else 1 for x in predictions]
submission['label'] = pd.DataFrame({'Prediction': new_predictions})

# 클러스터링 결과를 시각화한다
import matplotlib.pyplot as plt
plt.scatter(data[:, 0], data[:, 1], c=predictions)
plt.show()

#time
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

submission.to_csv(save_path+'submit_air_'+date+ '.csv', index=False)


# import 공기압축_xgboost as xgb
# from sklearn.datasets import make_blobs

# # 데이터를 생성한다
# X, y = make_blobs(n_samples=1000, centers=3, random_state=42)

# # XGBoost 모델을 생성한다
# xgb_model = xgb.XGBClassifier()

# # 모델을 학습시킨다
# xgb_model.fit(X)

# # 예측 결과를 얻는다
# y_pred = xgb_model.predict(X)

# # 클러스터링 결과를 시각화한다
# import matplotlib.pyplot as plt
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()