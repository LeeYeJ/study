#
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# 어절 단위 토큰화 (어절 단위로 자르는게 꼭 좋은 건 아님)
text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer() #분리
token.fit_on_texts([text]) # 텍스트를 실행시킨다. / 한문장이 아닐수도 있으니 리스트 형태로 받아들인다.

# 키밸류 형태
print(token.word_index) # 중복수가 많은 순으로 인덱스가 생성됨   {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
print(token.word_counts) # 중복된 횟수만큼 카운트가 된다  OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])

# 문자를 순서대로 수치로 바꿈 
x = token.texts_to_sequences([text])
print(x) # [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]  -> (1행 11열) /  수치만 보면 가치가 부여된다 따라서 원핫인코딩 해줘야됨

#원핫 
########## 1. to_categorical ########## 
# print(len(x)) # 왜 1이야

# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# print(x.shape) # (1, 11, 9)

# Tokenizer는 1부터 카운턴데 원핫해주면 0부터 카운트돼서 맨앞 열에 0이 생겼다. -> 앞에 0만 들어간 열을 지워주면 됨

# ######### 2. get_dummies ############# 1차원으로 받아들여야됨
# import pandas as pd

# # x = pd.get_dummies(np.array(x).reshape(11,)) # unhashable type: 'list' -> x의 텍스트가 리스트 형태이다 / Data must be 1-dimensional 겟더미는 1차원으로 받음 따라서 리쉐잎
# x = pd.get_dummies(np.array(x).ravel()) # ravel()는 numpy에서 flatten과 같은 것임
# print(x)
# print(x.shape) # (11,8)

#넘파이로 바꿔야되는데 왜 바꿔야할까 왜 리스트는 안될까 -> 다음 파일에 이유가 있어

########## 3. 사이킷런 onehot ########## 2차원으로 받아들여야됨
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
x = onehot_encoder.fit_transform(np.array(x).reshape(11,1)).toarray()
print(x)
print(x.shape) # (11,8)

