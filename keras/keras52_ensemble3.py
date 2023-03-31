# y값이 두개일때 머지에서 분기로 두 모델로 뺴줌

#1. 데이터
import numpy as np

# 잘라줄때 randomstate 맞춰서 잘라줘야 동일하게 잘림
x1_datasets = np.array([range(100),range(301,401)]) 
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)])
x3_datasets = np.array([range(201,301),range(511,611),range(1300,1400)]) 
print(x1_datasets.shape, x2_datasets.shape) # (2, 100) (3, 100)

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
x3 = x3_datasets.T
print(x1.shape,x2.shape,x3.shape) # (100, 2) (100, 3) (100, 3)

y1 = np.array(range(2001,2101)) #환율 
y2 = np.array(range(1001,1101)) #금리 

###########concatenate와 Concatenate의 차이##################
'''
Concatenate는 클래스이고 concatenate는 함수이다.

클래스는 입력과 목록을 연결하는 계층으로 ()안에는 axis -1이 디폴트로 들어간다.

함수는 안에 인풋값과 axis가 -1로 디폴트 들어간다.
'''


from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,x3_train,x3_test, \
y1_train,y1_test,y2_train,y2_test=train_test_split( # \ 길땐 이걸로 한줄임을 표시
    x1, x2,x3,y1,y2, train_size=0.7, random_state=546552
)
# y_train,y_test,x2_train,x2_test=train_test_split(
#     y, train_size=0.7, random_state=333
# )
print(x1_train.shape,x1_test.shape) # (70, 2) (30, 2)
print(x2_train.shape,x2_test.shape) # (70, 3) (30, 3)
print(x3_train.shape,x3_test.shape) # (70, 3) (30, 3)
print(y1_train.shape,y1_test.shape) # (70,) (30,)
print(y2_train.shape,y2_test.shape) # (70,) (30,)


#2모델
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input,Dropout

#2-1 모델1
input1 = Input(shape=(2,))
dense1 = Dense(10,activation='relu', name='stock1')(input1)
dense2 = Dense(20,activation='relu', name='stock2')(dense1)
Drop1 = Dropout(0.2)(dense2)
dense3 = Dense(30,activation='relu', name='stock3')(Drop1)
output1 = Dense(10, name='output1')(dense3) # 아웃풋 노드의 갯수는 자유롭게 줘도됨 오히려 적으면 많이 축소됨 모델 1,2는 합병된 전체 모델의 히든레이어임

#2-2 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10,name='weather1')(input2)
dense12 = Dense(10,name='weather2')(dense11)
dense13 = Dense(10,name='weather3',activation='relu')(dense12)
dense14 = Dense(10,name='weather4')(dense13)
output2 = Dense(10,name='output2')(dense14)

#2-3 모델3
input3 = Input(shape=(3,))
dense111 = Dense(10,activation='relu', name='stock11')(input3)
dense222 = Dense(20,activation='relu', name='stock22')(dense111)
Drop11 = Dropout(0.2)(dense222)
dense333 = Dense(30,activation='relu', name='stock33')(Drop11)
output3 = Dense(10, name='output3')(dense333) #

import tensorflow as tf
#2-4 머지 # 히든이니까 값 크게줘도됨
from tensorflow.keras.layers import concatenate ,Concatenate # 소문자는 함수/ 대문자 클래스 /concatenate 사슬 같이 잇다
merge1 = Concatenate()([output1, output2,output3])
# merge1 = concatenate([output1, output2,output3],name='mg1') # 두 모델의 아웃풋을 합병한다. / 두개 이상이니까 리스트 형태로 받는다.
merge2 = Dense(24, activation='relu', name='mg2')(merge1)
merge3 = Dense(36, activation='relu', name='mg3')(merge2)
hidden_output = Dense(1,name='hidden_output')(merge3)

#2-5 분기1

bungi1 =Dense(10,activation='selu',name='bg1')(hidden_output)
bungi2 =Dense(10 , name='bg2')(bungi1)
last_output1 =Dense(1)(bungi2)

#2-6 분기2

last_output2 = Dense(1, activation='linear',name='bungi11')(hidden_output)

model = Model(inputs=[input1,input2,input3],outputs =[last_output1,last_output2])

model.summary()

# 3. 컴파일 , 훈련
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss ='mse', optimizer='adam')

es = EarlyStopping(monitor='loss',mode='auto',patience=20,restore_best_weights=True)
model.fit([x1_train,x2_train,x3_train],[y1_train,y2_train], epochs=100,batch_size=1)

results = model.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])
print(results)

from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
y_predict = model.predict([x1_test,x2_test,x3_test])
print(y_predict) 
print(len(y_predict),len(y_predict[0])) #2 ,30 각각 30개씩 들어있음 #리스트는 shape로 확인 못함 그래서 len으로 확인


r2_1 = r2_score(y1_test,y_predict[0])
print("r2_1 스코어 : ", r2_1)

r2_2 = r2_score(y2_test,y_predict[1])
print("r2_2 스코어 : ", r2_2)

print('r2스코어 : ',(r2_1 + r2_2)/2)

def RMSE(a,b): 
    return np.sqrt(mean_squared_error(a,b)) 
rmse1 = RMSE(y1_test, y_predict[0])  
rmse2 = RMSE(y1_test, y_predict[1])                           
                         
print("RMSE : ", rmse1 , rmse2)


loss = model.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])
print('loss :', loss)

'''
r2 스코어 :  0.995119379607226
RMSE :  2.0603584278105735
1/1 [==============================] - 0s 131ms/step - loss: 4.2451
loss : 4.245076656341553
'''




# #1. 데이터
# import numpy as np

# # 잘라줄때 randomstate 맞춰서 잘라줘야 동일하게 잘림
# x1_datasets = np.array([range(100),range(301,401)]) 
# x2_datasets = np.array([range(101,201),range(411,511),range(150,250)])
# x3_datasets = np.array([range(201,301),range(511,611),range(1300,1400)]) 
# print(x1_datasets.shape, x2_datasets.shape) # (2, 100) (3, 100)

# x1 = np.transpose(x1_datasets)
# x2 = x2_datasets.T
# x3 = x3_datasets.T
# print(x1.shape,x2.shape,x3.shape) # (100, 2) (100, 3) (100, 3)

# y1 = np.array(range(2001,2101)) #환율 
# y2 = np.array(range(1001,1101)) #금리 


# from sklearn.model_selection import train_test_split
# x1_train,x1_test,x2_train,x2_test,x3_train,x3_test,y1_train,y1_test,y2_train,y2_test=train_test_split(
#     x1, x2,x3,y1,y2, train_size=0.7, random_state=546552
# )
# # y_train,y_test,x2_train,x2_test=train_test_split(
# #     y, train_size=0.7, random_state=333
# # )
# print(x1_train.shape,x1_test.shape) # (70, 2) (30, 2)
# print(x2_train.shape,x2_test.shape) # (70, 3) (30, 3)
# print(x3_train.shape,x3_test.shape) # (70, 3) (30, 3)
# print(y1_train.shape,y1_test.shape) # (70,) (30,)
# print(y2_train.shape,y2_test.shape) # (70,) (30,)


# #2모델
# from tensorflow.keras.models import Sequential,Model
# from tensorflow.keras.layers import Dense,Input,Dropout

# #2-1 모델1
# input1 = Input(shape=(2,))
# dense1 = Dense(10,activation='relu', name='stock1')(input1)
# dense2 = Dense(20,activation='relu', name='stock2')(dense1)
# Drop1 = Dropout(0.2)(dense2)
# dense3 = Dense(30,activation='relu', name='stock3')(Drop1)
# output1 = Dense(10, name='output1')(dense3) # 아웃풋 노드의 갯수는 자유롭게 줘도됨 오히려 적으면 많이 축소됨 모델 1,2는 합병된 전체 모델의 히든레이어임

# #2-2 모델2
# input2 = Input(shape=(3,))
# dense11 = Dense(10,name='weather1')(input2)
# dense12 = Dense(10,name='weather2')(dense11)
# dense13 = Dense(10,name='weather3',activation='relu')(dense12)
# dense14 = Dense(10,name='weather4')(dense13)
# output2 = Dense(10,name='output2')(dense14)

# #2-3 모델3
# input3 = Input(shape=(3,))
# dense111 = Dense(10,activation='relu', name='stock11')(input3)
# dense222 = Dense(20,activation='relu', name='stock22')(dense111)
# Drop11 = Dropout(0.2)(dense222)
# dense333 = Dense(30,activation='relu', name='stock33')(Drop11)
# output3 = Dense(10, name='output3')(dense333) #

# from tensorflow.keras.layers import concatenate ,Concatenate # 소문자는 함수/ 대문자 클래스 /concatenate 사슬 같이 잇다
# merge1 = concatenate([output1, output2,output3],name='mg1') # 두 모델의 아웃풋을 합병한다. / 두개 이상이니까 리스트 형태로 받는다.
# merge2 = Dense(24, activation='relu', name='mg2')(merge1)
# merge3 = Dense(36, activation='relu', name='mg3')(merge2)
# last_output = Dense(1,name='last')(merge3)

# #2-3 모델y1

# out4= Dense(last_output)
# dense121 = Dense(10,name='weather11')(out4)
# dense131 = Dense(10,name='weather22')(dense121)
# dense141 = Dense(10,name='weather33',activation='relu')(dense131)
# dense151 = Dense(10,name='weather44')(dense141)
# output4 = Dense(10,name='output4')(dense151)

# #2-4 모델y2

# out5 = Dense(last_output)
# dense122 = Dense(10,activation='relu', name='stock1111')(out5)
# dense233 = Dense(20,activation='relu', name='stock2222')(dense122)
# Drop111 = Dropout(0.2)(dense233)
# dense344 = Dense(30,activation='relu', name='stock3333')(Drop111)
# output5 = Dense(10, name='output5')(dense344) 

# model = Model(inputs=[input1,input2,input3],outputs =[output4,output5])

# # model.summary()

# from tensorflow.keras.callbacks import EarlyStopping
# model.compile(loss ='mse', optimizer='adam')
# es = EarlyStopping(monitor='loss',mode='auto',patience=20,restore_best_weights=True)
# model.fit([x1_train,x2_train,x3_train],y[y1_train,y2_train], epochs=1000,batch_size=1)

# from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
# y_predict = model.predict([x1_test,x2_test,x3_test])

# r2 = r2_score([y1_test,y2_test],y_predict)
# print("r2 스코어 : ", r2)

# def RMSE(y_test,y_predict): 
#     return np.sqrt(mean_squared_error([y1_test,y2_test],y_predict)) 
# rmse = RMSE([y1_test,y2_test], y_predict)                           
# print("RMSE : ", rmse)


# loss = model.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])
# print('loss :', loss)

# '''
# r2 스코어 :  0.995119379607226
# RMSE :  2.0603584278105735
# 1/1 [==============================] - 0s 131ms/step - loss: 4.2451
# loss : 4.245076656341553
# '''
