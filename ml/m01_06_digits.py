from sklearn.model_selection import train_test_split, cross_val_score, KFold
#사이킷런 로드 디짓
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,LSTM,Conv1D,Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

x,y=load_digits(return_X_y=True)
# x=datasets.data
# y=datasets.target

#######다중이니까 원핫인코딩해주기###########

y = to_categorical(y)
print(y) 
print(y.shape) #(1797, 10)

##########################################

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

x_train,x_test,y_train,y_test=train_test_split(
    x,y, shuffle=True, random_state=3338478, train_size=0.9, stratify=y
)

print(x_train.shape) #(1617, 64)
print(x_test.shape) #(180, 64)

# scaler= MinMaxScaler() # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

x_train= x_train.reshape(1617,8,8)
x_test= x_test.reshape(180,8,8)
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

model = RandomForestClassifier()

scores = cross_val_score(model,x,y,cv=5, n_jobs=-1) # cv = 5라고 써도 됨 / 위에서 정의해줘도 되고 /n_jobs=-1 최대 쓰는거임
print('ACC :',scores,'\n cross_val_score 평균 :',round(np.mean(scores),4))
'''
-kfold-
ACC : [0.81944444 0.725      0.84679666 0.88300836 0.78272981] 
 cross_val_score 평균 : 0.8114
'''

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

# es=EarlyStopping(monitor='val_loss',mode='auto',patience=20,restore_best_weights=True)

# model.fit(x_train,y_train,epochs=500,batch_size=50,validation_split=0.1,callbacks=[es])

# loss=model.evaluate(x_test,y_test)
# print('loss :', loss)

# y_pre=np.round(model.predict(x_test))
# acc=accuracy_score(y_test,y_pre)
# print('acc :', acc)

'''

'''


