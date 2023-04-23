from sklearn.model_selection import train_test_split, cross_val_score, KFold
#분류
#캐글, 따릉이, 디아벳 최대 업로드수 해서 등수까지 스냅샷 찍어서 금요일 주말 내내 제출
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold


x,y=fetch_covtype(return_X_y=True) # 인터넷에서 가져와서 내 로컬에 저장되는거임. 만약 엉키면(에러) 파일 경로 찾아서 직접 삭제해줘야됨 / 사이킷럭 삭제시 cmd 창에 uninstall

# 원핫인코딩
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y= y.reshape(-1,1)
y=encoder.fit_transform(y).toarray()

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
n_splits = 5 # 디폴트값 5
kfold = KFold(n_splits = n_splits, shuffle=True,random_state=123) 

#데이터 분리
x_train,x_test,y_train,y_test=train_test_split(
    x,y, shuffle=True, random_state=2000,train_size=0.9
)
# scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

model = RandomForestClassifier()

scores = cross_val_score(model,x,y,cv=5, n_jobs=-1) # cv = 5라고 써도 됨 / 위에서 정의해줘도 되고 /n_jobs=-1 최대 쓰는거임
print('ACC :',scores,'\n cross_val_score 평균 :',round(np.mean(scores),4))
'''
-kfold-
ACC : [0.57476141 0.72949063 0.66155488 0.68564224 0.64347429] 
 cross_val_score 평균 : 0.659
'''

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
# es=EarlyStopping(monitor='val_acc',mode='auto',patience=50)
# model.fit(x_train,y_train,epochs=5000,batch_size=5000,validation_split=0.1,callbacks=[es])

# results=model.evaluate(x_test,y_test)
# print('results :', results)

# y_pre= model.predict(x_test)

# y_test_acc=np.argmax(y_test, axis=1) 
# # print(y_test_acc) 
# y_pre=np.argmax(y_pre,axis=1)
# # print(y_pre) 

# acc=accuracy_score(y_pre,y_test_acc)   
# print('acc :', acc)

