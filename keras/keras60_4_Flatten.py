from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,LSTM, GRU,Bidirectional,Reshape,Embedding,Flatten
import numpy as np

# 시계열 데이터

#1. 데이터
docs = ['너무 재밋어요', '참 최고에요','참 잘 만든 영화네요', '추천하고 싶은 영화입니다.','한 번 더 보고 싶네요',
        '글세요','별로예요','생각보다 지루해요','연기가 어색해요','재미없어요','너무 재미없다','참 재밋네요','환희가 잘 생기긴 했어요','환희가 안해요'
        ]

# 긍정 1 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])

token = Tokenizer()
token.fit_on_texts(docs) # []리스트로 넣으면 문장 단위 

print(token.word_index) 
'''
{'참': 1, '너무': 2, '잘': 3, '환희가': 4, '재밋어요': 5, '최고에요': 6, '만든': 7, '영화네요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11, '한': 12, '번': 13, '더': 14, '보고': 15, '싶네요': 16, '글세요
': 17, '별로예요': 18, '생각보다': 19, '지루해요': 20, '연기가': 21, '어색해요': 22, '재미없어요': 23, '재미없다': 24, '재밋네요': 25, '생기긴': 26, '했어요': 27, '안해요': 28}
'''

print(token.word_counts) 
'''
OrderedDict([('너무', 2), ('재밋어요', 1), ('참', 3), ('최고에요', 1), ('잘', 2), ('만든', 1), ('영화네요', 1), ('추천하고', 1), ('싶은', 1), ('영화입니다', 1), ('한', 1), ('번', 1), ('더', 1), ('보고', 1), 
('싶네요', 1), ('글세요', 1), ('별로예요', 1), ('생각보다', 1), ('지루해요', 1), ('연기가', 1), ('어색해요', 1), ('재미없어요', 1), ('재미없다', 1), ('재밋네요', 1), ('환희가', 2), ('생기긴', 1), ('했어요', 
1), ('안해요', 1)])
'''

x = token.texts_to_sequences(docs)
print(type(x)) #<class 'list'>
print(x) #[[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20], [21, 22], [23], [2, 24], [1, 25], [4, 3, 26, 27], [4, 28]]

# 각 문장의 길이가 다르니까 패딩으로 맞춰준다 (앞쪽을 0으로 채움)
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) # padding='pre' 앞에부터 0을 채워주겠다. / maxlen=5가 아니고 4라면 [12, 13, 14, 15, 16]의 앞에 하나(12) 날아감
print(pad_x)
'''
[[ 0  0  0  2  5]
 [ 0  0  0  1  6]
 [ 0  1  3  7  8]
 [ 0  0  9 10 11]
 [12 13 14 15 16]
 [ 0  0  0  0 17]
 [ 0  0  0  0 18]
 [ 0  0  0 19 20]
 [ 0  0  0 21 22]
 [ 0  0  0  0 23]
 [ 0  0  0  2 24]
 [ 0  0  0  1 25]
 [ 0  4  3 26 27]
 [ 0  0  0  4 28]]
'''
print(pad_x.shape) # (14, 5)

# pad_x.reshape(14,5,1)
pad_x.reshape(pad_x.shape[0],pad_x.shape[1],1)

word_size = len(token.word_index)
print('단어 사전의 갯수 :',word_size) # 단어 사전의 갯수 : 28

#2. 모델
model= Sequential()
# model.add(Embedding(28,32)) # Flatten에 바로 붙이지 못한다. (n,n,10) 이니까 바로 못 붙임
model.add(Embedding(28,32,input_length=5)) #(n,5,10)이니까 Flatten 붙여줄수있음
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(10,activation='relu'))
model.add(Dense(16))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
model.fit(pad_x, labels , epochs=30, batch_size=8)

acc = model.evaluate(pad_x,labels)[1] # -> loss 와 acc 값
print('acc :', acc)

##########실습#####################
x_predict = '나는 성호가 정말 재미없다 너무 정말'


