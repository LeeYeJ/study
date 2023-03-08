from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

#데이터
datasets = load_boston()
x=datasets.data
y=datasets['target'] #이렇게 컬럼 명시로 데이터 표시해도 됨
print(x.shape,y.shape) # (506, 13) (506,)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=123, test_size=0.2
)

#모델구성
model=Sequential()
model.add(Dense(10,input_dim=13))
model.add(Dense(5, activation='relu')) 
model.add(Dense(4, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

#컴파일 훈련

model.compile(loss='mse', optimizer='adam')
hist= model.fit(x_train,y_train, epochs=500, batch_size=50, validation_split=0.2)
print(hist.history)

#로스값은 핏에 저장되어있다. model.fit은 결과치를 반환한다. 변수에 저장해서. -> 데이터 형태만 나옴 -> print(hist.history) -> 로스값들 나옴
'''
print(hist.history)

{'loss': [109.0726318359375, 62.594886779785156, 60.908512115478516, 58.67799377441406, 54.565155029296875, 52.15105438232422, 
50.705047607421875, 47.587806701660156, 44.04174041748047, 44.19512939453125, 39.54021072387695, 40.205039978027344, 35.90504837036133,
38.71308135986328, 35.09465789794922, 35.98184585571289, 31.126672744750977, 31.77972984313965, 32.280826568603516, 34.15589904785156], 
'val_loss': [73.31501007080078, 60.66437530517578, 60.19684982299805, 59.10980987548828, 55.461997985839844, 58.54871368408203, 

58.666744232177734, 58.214969635009766, 50.36850357055664, 49.47733688354492, 51.508819580078125, 53.301551818847656, 59.66644287109375, 
45.2325439453125, 45.05647277832031, 44.97794723510742, 50.82244110107422, 46.16643524169922, 41.485965728759766, 48.913639068603516]}
'''

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') # 순서대로 갈때는 x명시할 필요 없을
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss') #marker='.' 점점으로 표시->선이 됨
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('loss , val_loss')
plt.legend() # 선에 이름 표시
plt.grid() #격자 표시
plt.show()
#발로스가 더 중요하다고?




