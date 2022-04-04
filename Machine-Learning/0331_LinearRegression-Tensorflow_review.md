# Tensorflow


```python
import tensorflow as tf

# 상수 node 만들기
node1 = tf.constant(10, dtype=tf.float32)
node2 = tf.constant(30, dtype=tf.float32)

node3 = node1 + node2

# Session 필요 => 그래프 실행시키기 위해
sess = tf.Session()

print(sess.run(node1))  # graph 실행 -> node1 실행
print(sess.run(node3))  # graph 실행 -> node3 실행
print(sess.run([node1, node3]))
```

    10.0
    40.0
    [10.0, 40.0]
    


```python
# 현재 tensorflow 버전은 1.15버전

import tensorflow as tf

node1 = tf.placeholder(dtype=tf.float32)  # scalar
# node1 = tf.placeholder(shape=[2,2], dtype=tf.float32)  # 다차원 데이터일 경우

node2 = tf.placeholder(dtype=tf.float32)

node3 = node1 + node2

sess = tf.Session()

result = sess.run(node3, feed_dict={node1: 10,
                                    node2: 30})
print(result)
```

    40.0
    

## Tensorflow로 Multiple Linear Regression 구현하기


```python
import tensorflow as tf
import pandas as pd
import numpy as np


# Raw Data Loading
df = pd.read_csv('./data/student_exam_score.csv')
# display(df.head())

# 결측치 없고 이상치 없음, 정규화 필요 없음 => 데이터 전처리 X

# Training Data Set
x_data = df.drop('exam', axis=1, inplace=False)  # (25,3)
t_data = df['exam'].values.reshape(-1,1)  # (25,1)

# Prediction
predict_data = np.array([[90, 100, 95]])  # (1, 3)

# Placeholder
X = tf.placeholder(shape=[None,3], dtype=tf.float32)
T = tf.placeholder(shape=[None,1], dtype=tf.float32)

# Weight, bias
# W는 (3,1) 형태여야 하고 랜덤하게 생성
W = tf.Variable(tf.random.normal([3,1]))
b = tf.Variable(tf.random.normal([1]))

# model(hypothesis, 가설, predict model)   # y = Wx + b  => XW + b
H = tf.matmul(X,W) + b   

# loss function
loss = tf.reduce_mean(tf.square(H-T))

# train node를 생성해야 함
train = tf.train.GradientDescentOptimizer(learning_rate=1e-7).minimize(loss)

# Session 생성하고 초기화 진행
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # 초기화 구문
# (그래프를 실행하기 전에 먼저 실행되어야 함)

# 반복 학습
for step in range(300000):
    
    _, W_val, b_val, loss_val = sess.run([train, W, b, loss], 
                                         feed_dict={X: x_data,
                                                    T: t_data})
    if step % 30000 == 0:
        print('W : {}, b : {}, loss : {}'.format(W_val, b_val, loss_val))
```

    
    W : [[ 1.5190185]
     [ 1.2637861]
     [-1.169783 ]], b : [0.62325746], loss : 1474.7386474609375
    W : [[ 1.5677472]
     [ 1.2686144]
     [-0.779564 ]], b : [0.62685335], loss : 112.79974365234375
    W : [[ 1.4759965]
     [ 1.1447824]
     [-0.569425 ]], b : [0.6286415], loss : 90.11665344238281
    W : [[ 1.3914781 ]
     [ 1.0386504 ]
     [-0.38356382]], b : [0.6291849], loss : 72.42874145507812
    W : [[ 1.3135942 ]
     [ 0.9478372 ]
     [-0.21913366]], b : [0.6291849], loss : 58.612606048583984
    W : [[ 1.2415763 ]
     [ 0.87036663]
     [-0.07342018]], b : [0.6291849], loss : 47.78836441040039
    W : [[1.1752849 ]
     [0.8043048 ]
     [0.05554282]], b : [0.6291849], loss : 39.30393600463867
    W : [[1.1138297 ]
     [0.7482958 ]
     [0.17000726]], b : [0.6291849], loss : 32.623863220214844
    W : [[1.0573883 ]
     [0.70069104]
     [0.271381  ]], b : [0.6291849], loss : 27.366567611694336
    W : [[1.0051597]
     [0.6605492]
     [0.361359 ]], b : [0.6291849], loss : 23.210588455200195
    

### 학습 종료 후 예측하기


```python
result = sess.run(H, 
                  feed_dict={X: np.array([[89, 100, 95]])})
print(result)
```

    [[190.39206]]
    
