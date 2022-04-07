# Binary Classification
- 위스콘신 유방암 데이터

## 1. sklearn 구현 


```python
import numpy as np
from sklearn import linear_model  # LogisticRegression()
from sklearn.datasets import load_breast_cancer       # 데이터 로딩 위한 함수
from sklearn.model_selection import train_test_split  # 학습데이터와 평가데이터 분리
from sklearn.model_selection import cross_val_score   # cross validation
```


```python
# Raw Data Loading
cancer = load_breast_cancer()
# print(type(cancer))  # <class 'sklearn.utils.Bunch'>
                       # sklearn이 데이터를 표현하기 위해 사용하는 자료구조
                       # python의 dictionary와 유사한 구조
# print(cancer)      
# data와 target이라는 속성을 가짐
# data : 독립변수, target : 종속변수
# print(cancer.data.shape, cancer.target.shape)  # (569, 30) (569,)

# print(np.unique(cancer.target, return_counts=True))  # array([0, 1]), array([212, 357]
# print(cancer.DESCR)  # 유방암 데이터에 대한 상세 내용
# :Missing Attribute Values: None
# :Class Distribution: 212 - Malignant(악성), 357 - Benign(정상)
```


```python
# Data Set
x_data = cancer.data
t_data = cancer.target

# Hold-out validation을 위해 train과 validation 데이터 분리
train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(x_data,
                 t_data,
                 test_size=0.2,    # 지정하지 않을 경우 train: 75%, test: 25%
                 random_state=2,
                 stratify=t_data)  # t_data 안에 있는 class의 비율대로 데이터를 나누겠다는 속성
# print(train_x_data.shape, train_t_data.shape)  # (455, 30) (455,)
# print(np.unique(train_t_data, return_counts=True)) 
# array([0, 1]), array([170, 285])  # 0과 1의 각 개수

# Model 생성
model = linear_model.LogisticRegression()
```


```python
# K-Fold cross validation
test_score = cross_val_score(model, x_data, t_data, scoring='accuracy', cv=5)  # scoring : metric, cv : k-fold의 k값
print(test_score)  # 각 fold의 accuracy
print(test_score.mean())
```

    [0.92982456 0.93859649 0.95614035 0.9122807  0.95575221]
    0.9385188635305075
    


```python
# Hold-out 방식으로 validation 
model.fit(train_x_data, train_t_data) 
test_score = model.score(test_x_data, test_t_data)
print(test_score)
```

    0.9736842105263158
    

## 2. Tensorflow 구현


```python
import tensorflow as tf

# tensorflow 그래프 그리기

# placeholder
X = tf.placeholder(shape=[None,30], dtype=tf.float32)
T = tf.placeholder(shape=[None,1], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([30,1]))
b = tf.Variable(tf.random.normal([1]))

# Hypothesis, model, predict model, Logistic Regression Model
logit = tf.matmul(X,W) + b
H = tf.sigmoid(logit)

# cross entropy(loss function)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,
                                                              labels=T))

# train 
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

# Session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # 초기화 작업

# 반복학습
# 전체 데이터를 이용해서 1번 학습 => 1 epoch(에폭)
for step in range(100000):
    _, loss_val = sess.run([train, loss], feed_dict={X: train_x_data,
                                                     T: train_t_data.reshape(-1,1)})
#     + reg_strength* tf.nn._l2_loss(W)  # 규제하는 경우

    if step % 10000 == 0:
        print('loss value : {}'.format(loss_val))
```
  
  
    loss value : 187.8224639892578
    loss value : 0.4163092076778412
    loss value : 0.40795615315437317
    loss value : 0.40019702911376953
    loss value : 0.392855167388916
    loss value : 0.3859238624572754
    loss value : 0.37935081124305725
    loss value : 0.3730590343475342
    loss value : 0.36699122190475464
    loss value : 0.36106348037719727
    


```python
# 정확도(accuracy) 측정

# validation data(test_x_data, test_t_data)를 이용해서 정확도 측정
predict = tf.cast(H >= 0.5, dtype=tf.float32)  # True -> 1.0
                                               # False -> 0.0 
correct = tf.equal(predict, T)  # True, False, False, True, ...  
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))  #   1  , 0   ,  0   ,  1

accuracy_val = sess.run(accuracy, feed_dict={X: test_x_data,
                                             T: test_t_data.reshape(-1,1)})
print('Accuracy : {}'.format(accuracy_val))
```

    Accuracy : 0.8859649300575256
    
