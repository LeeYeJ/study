from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

#데이터
datasets = load_boston()
x=datasets.data
y=datasets['target'] #이렇게 컬럼 명시로 데이터 표시해도 됨
print(x.shape,y.shape) # (506, 13) (506,)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,shuffle=True,random_state=650, test_size=0.2
)

#모델구성
model=Sequential()
model.add(Dense(10,input_dim=13))
model.add(Dense(5)) # 0-1 사이 한정
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(1))

#컴파일 훈련

model.compile(loss='mse', optimizer='adam') 

from tensorflow.python.keras.callbacks import EarlyStopping
#정의
es=EarlyStopping(monitor='val_loss',patience=30,mode='min',verbose=1,restore_best_weights=True) #monitor='val_loss'를 기준으로 한다, 최소값을 찾았을때 그때부터 patience=10 참는 횟수
#mode=min, max, auto -> auto는 자동으로 맞춰줌 최소값 찾을건지 최댓값 찾을건지 , verbose=1 끊기는 지점 확인 가능
#restore_best_weights=True -> 이걸 해줘야 patience전의 최소값의 웨이트값을 가져올수있다.

hist= model.fit(x_train,y_train, epochs=1000, batch_size=30, validation_split=0.2,verbose=1,callbacks=[es]) #callbacks=[es]호출

# print(hist) #<tensorflow.python.keras.callbacks.History object at 0x0000015BC965E250>
# print('==========================')
# print(hist.history)
# '''
# {'loss': [661.46142578125, 586.4367065429688, 531.8495483398438, 479.5317687988281, 412.90557861328125, 
# 332.769775390625, 263.0555725097656, 212.4530792236328, 193.12322998046875, 186.6245574951172], 

# 'val_loss': [605.1642456054688, 545.0879516601562, 494.89654541015625, 432.573486328125, 355.6365966796875, 
# 277.7493896484375, 214.8768310546875, 182.39605712890625, 175.0891571044922, 168.51902770996094]}
# '''
# print('==========================')
# print(hist.history['loss'])
# '''
# [219.62176513671875, 184.17086791992188, 171.16598510742188, 153.6690673828125, 143.3023681640625, 
# 135.28530883789062, 127.63369750976562, 121.48835754394531, 115.06039428710938, 109.5584945678711]
# '''
# print('==========================')
# print(hist.history['val_loss'])
'''
[173.7381591796875, 164.6244659423828, 147.9994659423828, 133.49586486816406, 125.65687561035156, 
118.78868865966797, 114.18157958984375, 108.2010269165039, 103.2557373046875, 98.12283325195312]
'''

loss=model.evaluate(x_test,y_test)
print('loss:', loss)

y_pre=model.predict(x_test)
r2=r2_score(y_pre,y_test)
print('r2스코어:', r2)



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
import matplotlib
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글깨짐 해결 #다르 폰트 필요하면 윈도우 폰트 파일에 추가해줘야됨 # 상용할땐 나눔체로 쓰자.
plt.figure(figsize=(9,6)) #그래프의 사이즈 , 단위는 inch
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') # 순서대로 갈때는 x명시할 필요 없을
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss') #marker='.' 점점으로 표시->선이 됨 
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('loss , val_loss')
plt.legend() # 선에 이름 표시 ,범례
plt.grid() #격자 표시
plt.show()

#발로스가 더 중요하다고?

# 클래스 함수 차이 -> 분량은 반이상 차게 이메일 제출

'''
Epoch 725/1000
11/11 [==============================] - 0s 3ms/step - loss: 28.5732 - val_loss: 21.6936
Epoch 726/1000
11/11 [==============================] - 0s 3ms/step - loss: 27.6175 - val_loss: 23.6604
Epoch 727/1000
11/11 [==============================] - 0s 3ms/step - loss: 32.3204 - val_loss: 22.4820
Restoring model weights from the end of the best epoch.
Epoch 00727: early stopping
4/4 [==============================] - 0s 997us/step - loss: 19.7138
loss: 19.713762283325195
r2스코어: 0.6425158227355681
'''
