# 파라미터 알고싶으면 keras.io 들어가서 API docs 들어가서 확인 tensorflow.org도 들어가서 확인 가능
# kaggle.com에서 데이터나 대회 확인 가능 문풀 가능
# https://dacon.io/ 국내 대회 페북으로 가입함.. 구글 저장공간없음ㅠ
# RMSE 제곱에 루트씌운거




#보스턴에 있는 집값을 찾는 지표
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
# 데이터부분은 x데이터
# 타겟부분은 y데이터

x_train, x_test, y_train, y_test = train_test_split(x, y, #x는 x_train과 x_test로 분리되고, y는 y_train과 y_test 순서로! 분리된다.
     train_size=0.7, shuffle=True, random_state=1188    # 행 506개에서 train 0.7 사이즈니까 1 epochs당 354번 돌음
)

#random_state= 7995,45681

#print(x)
#print(y)
#print(datasets)
#print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

#print(datasets.DESCR)  데이터셋의 정보

#print(x.shape, y.shape) #(506, 13) (506,)


#2.모델 구성
model=Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=3000, batch_size=100 , verbose=2) # verbose=1이 디폴트값인데 0으로하면 결과만 보여줌 ,2로 해주면 터미널에 딜레이되는 진행바가 안보임 3 이상은 에포만 나옴

'''
verbose
 0은 아무것도 안보여줌
 1, auto는 다 보여줘
 2는 프로그래스바만 없어져
 3,4,5... 나머지는 에포만 나온다
'''

'''
#4.평가 예측
loss=model.evaluate(x_test,y_test,verbose=0)
print('loss : ', loss)

y_predict=model.predict(x_test) # 훈련 안시킨 데이터에서 예측하자 아래

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # 원값과 예측값이 얼마나 맞는지 확인할 수 있다. / 얘도 훈련안한 y_test로 확인해보자 (내신,수능 비교)
print('r2스코어 :', r2) # 값은 1과 가까울 수록 좋다.
'''

