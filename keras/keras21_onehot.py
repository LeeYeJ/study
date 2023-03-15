#[과제]

#3가지 원핫인코딩 방식을 비교할 것

#1. pandas의 get_dummies
'''
y = pd.get_dummies(y)
y = np.array(y) # type 바꿔줘
print(y.shape)

'''

#2. keras의 to_categorical
'''
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
print(y.shape)

y밸류가 0부터 시작하는걸로 써주는데 아니라면(예를들어 1) 
0부터 시작하게 해서 그 컬럼에 다 0을 넘어주고 컬럼 삭제를 해주자
'''

#3. sklean의 OneHotEncoder

'''
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y= y.reshape(-1,1)
y=encoder.fit_transform(y).toarray()
'''

# 미세한 차이를 정리하세요.
