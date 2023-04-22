

#https://dacon.io/competitions/open/236068/mysubmission?isSample=1
#당뇨대회

from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold


path='./_data/dacon_diabets/'
path_save='./_save/dacon_diabets/'

train_csv=pd.read_csv(path+'train.csv',index_col=0) #@@인덱스!

test_csv=pd.read_csv(path+'test.csv',index_col=0)

print(train_csv.isnull().sum()) #@@@@@@@@@결측치 확인!
'''
ID                          0
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
'''
#@@@@@@@@@데이터분리
x=train_csv.drop(['Outcome'],axis=1) #@@@@
y=train_csv['Outcome']
print(x.columns)
print(y)
print(x.shape) # (652, 8)
print(y.shape) # (652,)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=4545451,train_size=0.9,
)

# scaler= MinMaxScaler(
    # ) # StandardScaler 써줄거면 민맥스 대신 StandardScaler() 써주면 끝
# scaler= StandardScaler()
# scaler= RobustScaler()
scaler= MaxAbsScaler()
scaler.fit(x_train) # fit의 범위가 x_train이다 
x_train=scaler.transform(x_train) #변환시키라
x_test=scaler.transform(x_test)

#test 파일도 스케일링 해줘야됨!!!!!!!!!
test_csv=scaler.transform(test_csv)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(x_train,y_train,epochs=500,batch_size=5,validation_split=0.1)


# results=model.evaluate(x_test,y_test)
# print('results:', results)

y_pre=np.round(model.predict(x_test))
acc=accuracy_score(y_pre,y_test)
print('acc:',acc)

result = model.score(x,y)
print(result) 

y_sub=np.round(model.predict(test_csv))
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
submission['Outcome']=y_sub

submission.to_csv(path_save+'submisssion_03131820.csv')







