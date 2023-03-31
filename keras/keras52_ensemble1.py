# 대회때 프레딕 데이터로 모델 한번 돌려서 그걸 트레인 데이터에 넣어서 다시 모델 돌려서 사용하기도 함

#1. 데이터
import numpy as np

x1_datasets = np.array([range(100),range(301,401)]) # 삼성 ,아모레
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]) #온도, 습도, 강수
print(x1_datasets.shape, x2_datasets.shape) # (2, 100) (3, 100)

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
print(x1.shape,x2.shape) # (100, 2) (100, 3)

y = np.array(range(2001,2101)) #환율

from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(
    x1, x2,y, train_size=0.7, random_state=546552
)
# y_train,y_test,x2_train,x2_test=train_test_split(
#     y, train_size=0.7, random_state=333
# )
print(x1_train.shape,x1_test.shape) #(70, 2) (30, 2)
print(x2_train.shape,x2_test.shape) #(70, 3) (30, 3)
print(y_train.shape,y_test.shape) #(70,) (30,)

#2모델
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input,Dropout

#2-1 모델1
input1 = Input(shape=(2,))
dense1 = Dense(10,activation='relu', name='stock1')(input1)
dense2 = Dense(20,activation='relu', name='stock2')(dense1)
Drop1 = Dropout(0.2)(dense2)
dense3 = Dense(30,activation='relu', name='stock3')(Drop1)
output1 = Dense(1, name='output1')(dense3) # 아웃풋 노드의 갯수는 자유롭게 줘도됨 오히려 적으면 많이 축소됨 모델 1,2는 합병된 전체 모델의 히든레이어임

#2-2 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10,name='weather1')(input2)
dense12 = Dense(10,name='weather2')(dense11)
dense13 = Dense(10,name='weather3',activation='relu')(dense12)
dense14 = Dense(10,name='weather4')(dense13)
output2 = Dense(1,name='output2')(dense14)

from tensorflow.keras.layers import concatenate ,Concatenate # 소문자는 함수/ 대문자 클래스 /concatenate 사슬 같이 잇다
merge1 = concatenate([output1, output2],name='mg1') # 두 모델의 아웃풋을 합병한다. / 두개 이상이니까 리스트 형태로 받는다.
merge2 = Dense(2, activation='relu', name='mg2')(merge1)
merge3 = Dense(3, activation='relu', name='mg3')(merge2)
last_output = Dense(1,name='last')(merge3)

model = Model(inputs=[input1,input2],outputs = last_output)


# model.summary()

from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss ='mse', optimizer='adam')
es = EarlyStopping(monitor='loss',mode='auto',patience=20,restore_best_weights=True)
model.fit([x1_train,x2_train],y_train, epochs=500,batch_size=1)

from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
y_predict = model.predict([x1_test,x2_test])

r2 = r2_score(y_test,y_predict)
print("r2 스코어 : ", r2)

def RMSE(y_test,y_predict): 
    return np.sqrt(mean_squared_error(y_test,y_predict)) 
rmse = RMSE(y_test, y_predict)                           
print("RMSE : ", rmse)


loss = model.evaluate([x1_test,x2_test],y_test)
print('loss :', loss)
'''
'''