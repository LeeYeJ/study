from sklearn.model_selection import train_test_split, cross_val_score, KFold,cross_val_predict
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import pandas as pd

#데이터 불러오기
path = './_data/kaggle_bike/'
path_save='./_save/kaggle_bike/'
train_csv= pd.read_csv(path +'train.csv', index_col=0)
test_csv=pd.read_csv(path + 'test.csv', index_col=0)

#데이터 결측치 제거
print(train_csv.isnull().sum()) # 결측값 없음

#데이터 x,y분리
x=train_csv.drop(['count','casual','registered'], axis=1)
print(x.columns)
'''
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed'], 컬럼 8개 'casual','registered'는 테스트 데이터에 없으니까 그냥 일단 삭제해보자
'''
y=train_csv['count']
print(y)

kf = KFold(n_splits=5,shuffle=True, random_state=3324)

# 데이터 split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=4481,train_size=0.9
)
print(x_train.shape,x_test.shape) # (9797, 8) (1089, 8)
print(y_train.shape,y_test.shape) # (9797,) (1089,)

# 모델 구성
model = RandomForestRegressor()


scores = cross_val_score(model,x_train,y_train,cv=5, n_jobs=-1) # cv = 5라고 써도 됨 / 위에서 정의해줘도 되고 /n_jobs=-1 최대 쓰는거임
y_pred = cross_val_predict(model,x_test,y_test,cv =kf)

r2 = r2_score(y_test,y_pred)
print('r2_score :',scores,'\n cross_val_score 평균 :',round(np.mean(scores),4))
print('cross_val_predict_R2 :',r2)
# #컴파일 훈련
# model.compile(loss='mae',optimizer='adam')
# model.fit(x_train,y_train,epochs=500,batch_size=200,validation_split=0.2)

# #평가
# loss= model.evaluate(x_test,y_test)
# print('loss :',loss)

# #예측 r2스코어 확인
# y_pre=model.predict(x_test)
# r2=r2_score(y_test,y_test)
# print('r2 스코어 :', r2)

# # RMSE 함수 정의
# def RMSE(y_test,y_pre):
#     return np.sqrt(mean_squared_error(y_test,y_pre)) #정의
# rmse=RMSE(y_test,y_pre) #사용
# print('RMSE :',rmse)

#카운트값 빼기
y_submit = model.predict(test_csv)
print(y_submit)

#카운트값 넣어주기
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)
print(submission)

#제출파일의 count에 y_submit을 넣어준다.
submission['count'] = y_submit
print(submission)

submission.to_csv(path_save + 'submit_0307_2046.csv')
'''
Epoch 499/500
40/40 [==============================] - 0s 2ms/step - loss: 108.0786 - val_loss: 111.9325
Epoch 500/500
40/40 [==============================] - 0s 1ms/step - loss: 108.0147 - val_loss: 111.8740
35/35 [==============================] - 0s 602us/step - loss: 110.8073
loss : 110.80728149414062
r2 스코어 : 1.0
RMSE : 156.2863592601946

-
'''


















model = RandomForestClassifier()

scores = cross_val_score(model,x,y,cv=5, n_jobs=-1) # cv = 5라고 써도 됨 / 위에서 정의해줘도 되고 /n_jobs=-1 최대 쓰는거임
print('ACC :',scores,'\n cross_val_score 평균 :',round(np.mean(scores),4))