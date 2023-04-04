#59_1 카피
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# 어절 단위 토큰화 (어절 단위로 자르는게 꼭 좋은 건 아님)
text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '나는 지구용사 배환희다. 멋있다. 또 또 얘기해부아'

token = Tokenizer() #분리
token.fit_on_texts([text,text2]) # 텍스트를 실행시킨다. / 한문장이 아닐수도 있으니 리스트 형태로 받아들인다.

# 키밸류 형태
print(token.word_index)
'''
{'마구': 1, '나는': 2, '매우': 3, '또': 4, '진짜': 5, '맛있는': 6, '밥을': 7, '엄청': 8, '먹었다': 9, '지구용사': 10, '배환희다': 11, '멋있다': 12, '얘기해부아': 13}
'''
print(token.word_counts) 
'''
OrderedDict([('나는', 2), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1), ('지구용사', 1), ('배환희다', 1), ('멋있다', 1), ('또', 2), ('얘기해부아', 1)])
'''
# 문자를 순서대로 수치로 바꿈 
x = token.texts_to_sequences([text,text2])
print(type(x)) # <class 'list'>
print(x) # [[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9], [2, 10, 11, 12, 4, 4, 13]]

x = x[0] + x[1]
print(x)

#원핫 
########## 1. to_categorical ########## 
# print(len(x)) # 왜 1이야

# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# print(x.shape) # (18, 14) -> 원래는 13임. 0부터 시작돼서 0이 추가 되어 14된거임

# Tokenizer는 1부터 카운턴데 원핫해주면 0부터 카운트돼서 맨앞 열에 0이 생겼다. -> 앞에 0만 들어간 열을 지워주면 됨

######### 2. get_dummies ############# 1차원으로 받아들여야됨
# import pandas as pd

# # x = pd.get_dummies(np.array(x).reshape(11,)) # unhashable type: 'list' -> x의 텍스트가 리스트 형태이다 / Data must be 1-dimensional 겟더미는 1차원으로 받음 따라서 리쉐잎
# x = pd.get_dummies(np.array(x).ravel()) # ravel()는 numpy에서 flatten과 같은 것임
# print(x)
# print(x.shape) # (18, 13)

#넘파이로 바꿔야되는데 왜 바꿔야할까 왜 리스트는 안될까 -> 다음 파일에 이유가 있어

########## 3. 사이킷런 onehot ########## 2차원으로 받아들여야됨
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
x = onehot_encoder.fit_transform(np.array(x).reshape(18,1)).toarray()
print(x)
print(x.shape) # (11,8)

# 붙일때
# 리스트는 append
# 넘파이는 concat

