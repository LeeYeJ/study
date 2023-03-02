#궁금한점
# 데이터에서 x의 차원과 y의 차원의 관계 


import numpy as np # 사람의 연산과 비슷하게 행렬 연산
from tensorflow.keras.models import Sequential # 순차적 모델
from tensorflow.keras.layers import Dense # 단순연산

#1. 데이터
x= np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
y= np.array([[1,2],[3,4]])

model=Sequential()
model.add(Dense(3,input_dim=2))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))

model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=1000,batch_size=1)

loss = model.evaluate(x,y)
print('loss :', loss)

result = model.predict([[[1,2],[3,4]]]) 
print('[[[1,2],[3,4]]]의 값은 : ', result)

#loss : 0.500485897064209
#WARNING:tensorflow:Model was constructed with shape (None, 2) for input KerasTensor(type_spec=TensorSpec(shape=(None, 2), dtype=tf.float32, name='dense_input'), name='dense_input', description="created by layer 'dense_input'"), but it was called on an input with incompatible shape (None, 2, 2).
#1/1 [==============================] - 0s 141ms/step
#[[[1,2],[3,4]]]의 값은 :  [[[0.996184  1.0021704]
#  [1.3321608 1.35117  ]]]



