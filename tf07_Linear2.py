# 1번에서 데이터만 다르게
import tensorflow as tf
tf.set_random_seed(337)

# 1. 데이터
x = [1,2,3,4,5]
y = [2,4,6,7,10]

w = tf.Variable(333, dtype=tf.float32) # 2에 가까워진다
b = tf.Variable(111, dtype=tf.float32) # 0에 가까워지고

####### 실습 맹그러 ############

# y를 찾는 부분
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # hypothesis - y 의 제곱의 평균 -> mse (loss지표)

# 웨이트 빼기 고정된 러닝레이트와 로스를 웨이트로 편미분 해준값을 곱한다 -> 값이 계속 작아진다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 경사하강법 방식으로 optimizer를 최적화 시켜준다.
train = optimizer.minimize(loss) # 따라서 로스를 최저로 뽑음 
# model.complie(loss='mse', optimizer = 'sgd') 이것과 같다?

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) # 전체 변수 초기화

#4 모델의 훈련
epochs = 2001
for step in range(epochs):
    sess.run(train) # 에포당 그래프가 한번에 모두 연산됨
    if step %20 ==0:
        print(step,sess.run(loss), sess.run(w), sess.run(b))

sess.close()
