# tensorflow 1.15 버전으로 구현하기 😌

## AND GATE 연산


```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Training Data Set
x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float64)

# AND GATE 연산에 대한 t_data
t_data = np.array([0, 0, 0, 1], dtype=np.float64)

# placeholder
X = tf.placeholder(shape=[None, 2], dtype=tf.float32)
T = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(tf.random.normal([1]))

# Hypothesis, model
logit = tf.matmul(X, W) + b
H = tf.sigmoid(logit)

# loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,
                                                              labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

# Session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 반복학습
for step in range(30000):
    _, loss_val = sess.run([train, loss],
                           feed_dict={X: x_data,
                                      T: t_data.reshape(-1,1)})
    if step % 3000 == 0:
        print(f'loss value: {loss_val}')
```


    loss value: 1.1210260391235352
    loss value: 0.28219640254974365
    loss value: 0.19526302814483643
    loss value: 0.14986202120780945
    loss value: 0.12145474553108215
    loss value: 0.10192770510911942
    loss value: 0.08767993748188019
    loss value: 0.07683644443750381
    loss value: 0.0683172419667244
    loss value: 0.06145477294921875
    


```python
# evaluation (모델 평가)
predict = tf.cast(H >= 0.5, dtype=tf.float32)
predict_val = sess.run(predict, feed_dict={X:x_data})
print(predict_val)

# classification_report(정답데이터(1차원), 예측데이터(1차원))
print(classification_report(t_data, predict_val.ravel()))
# 결과를 확인해서 logistic regression이 진리표를 학습할 수 있는지 확인
```

    [[0.]
     [0.]
     [0.]
     [1.]]
                  precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00         3
             1.0       1.00      1.00      1.00         1
    
        accuracy                           1.00         4
       macro avg       1.00      1.00      1.00         4
    weighted avg       1.00      1.00      1.00         4
    
    

## OR GATE 연산


```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Training Data Set
x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float64)

# OR GATE 연산에 대한 t_data
t_data = np.array([0, 1, 1, 1], dtype=np.float64)

# placeholder
X = tf.placeholder(shape=[None, 2], dtype=tf.float32)
T = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(tf.random.normal([1]))

# Hypothesis, model
logit = tf.matmul(X, W) + b
H = tf.sigmoid(logit)

# loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,
                                                              labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

# Session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 반복학습
for step in range(30000):
    _, loss_val = sess.run([train, loss],
                           feed_dict={X: x_data,
                                      T: t_data.reshape(-1,1)})
    if step % 3000 == 0:
        print(f'loss value: {loss_val}')
```

    loss value: 0.5036659240722656
    loss value: 0.2118951976299286
    loss value: 0.1347322165966034
    loss value: 0.09739359468221664
    loss value: 0.07569998502731323
    loss value: 0.061652518808841705
    loss value: 0.05187079310417175
    loss value: 0.04469473659992218
    loss value: 0.03921923041343689
    loss value: 0.03491150587797165
    


```python
# evaluation (모델 평가)
predict = tf.cast(H >= 0.5, dtype=tf.float32)
predict_val = sess.run(predict, feed_dict={X:x_data})
print(predict_val)

# classification_report(정답데이터(1차원), 예측데이터(1차원))
print(classification_report(t_data, predict_val.ravel()))
# 결과를 확인해서 logistic regression이 진리표를 학습할 수 있는지 확인
```

    [[0.]
     [1.]
     [1.]
     [1.]]
                  precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00         1
             1.0       1.00      1.00      1.00         3
    
        accuracy                           1.00         4
       macro avg       1.00      1.00      1.00         4
    weighted avg       1.00      1.00      1.00         4
    
    

## XOR GATE 연산


```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Training Data Set
x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float64)

# XOR GATE 연산에 대한 t_data
t_data = np.array([0, 1, 1, 0], dtype=np.float64)

# placeholder
X = tf.placeholder(shape=[None, 2], dtype=tf.float32)
T = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(tf.random.normal([1]))

# Hypothesis, model
logit = tf.matmul(X, W) + b
H = tf.sigmoid(logit)

# loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,
                                                              labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

# Session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 반복학습
for step in range(30000):
    _, loss_val = sess.run([train, loss],
                           feed_dict={X: x_data,
                                      T: t_data.reshape(-1,1)})
    if step % 3000 == 0:
        print(f'loss value: {loss_val}')
```

    loss value: 1.4369406700134277
    loss value: 0.6953383684158325
    loss value: 0.693306028842926
    loss value: 0.693161129951477
    loss value: 0.6931484937667847
    loss value: 0.6931473016738892
    loss value: 0.6931471824645996
    loss value: 0.6931471824645996
    loss value: 0.6931471824645996
    loss value: 0.6931471824645996
    


```python
# evaluation (모델 평가)
predict = tf.cast(H >= 0.5, dtype=tf.float32)
predict_val = sess.run(predict, feed_dict={X:x_data})
print(predict_val)

# classification_report(정답데이터(1차원), 예측데이터(1차원))
print(classification_report(t_data, predict_val.ravel()))
# 결과를 확인해서 logistic regression이 진리표를 학습할 수 있는지 확인
# => 학습이 안 됨
```

    [[0.]
     [0.]
     [0.]
     [1.]]
                  precision    recall  f1-score   support
    
             0.0       0.33      0.50      0.40         2
             1.0       0.00      0.00      0.00         2
    
        accuracy                           0.25         4
       macro avg       0.17      0.25      0.20         4
    weighted avg       0.17      0.25      0.20         4
    
    

# DNN으로 XOR GATE 연산을 학습할 수 있는지 확인하기 😐


```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Training Data Set
x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float64)

# XOR GATE 연산에 대한 t_data
t_data = np.array([0, 1, 1, 0], dtype=np.float64)

# placeholder
X = tf.placeholder(shape=[None, 2], dtype=tf.float32)
T = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Weight & bias
W2 = tf.Variable(tf.random.normal([2, 10]))  # hidden layer에 logistic이10개
b2 = tf.Variable(tf.random.normal([10]))
layer2 = tf.sigmoid(tf.matmul(X, W2) + b2)   # layer2에서 나가는 값

W3 = tf.Variable(tf.random.normal([10, 6]))  # hidden layer에 logistic이 6개
b3 = tf.Variable(tf.random.normal([6]))
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)   # layer3에서 나가는 값

W4 = tf.Variable(tf.random.normal([6, 1]))  # hidden layer에 logistic이 1개
b4 = tf.Variable(tf.random.normal([1]))


# Hypothesis, model
logit = tf.matmul(layer3, W4) + b4
H = tf.sigmoid(logit)

# loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,
                                                              labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

# Session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 반복학습
for step in range(30000):
    _, loss_val = sess.run([train, loss],
                           feed_dict={X: x_data,
                                      T: t_data.reshape(-1,1)})
    if step % 3000 == 0:
        print(f'loss value: {loss_val}')
```

    loss value: 1.389522671699524
    loss value: 0.6774378418922424
    loss value: 0.6457579135894775
    loss value: 0.5692681670188904
    loss value: 0.41968417167663574
    loss value: 0.250230073928833
    loss value: 0.1402740478515625
    loss value: 0.08446931093931198
    loss value: 0.05615539103746414
    loss value: 0.040459271520376205
    


```python
predict = tf.cast(H >= 0.5, dtype=tf.float32)
predict_val = sess.run(predict, feed_dict={X:x_data})
print(predict_val)

# classification_report(정답데이터(1차원), 예측데이터(1차원))
print(classification_report(t_data, predict_val.ravel()))
# 결과를 확인해서 logistic regression이 진리표를 학습할 수 있는지 확인
```

    [[0.]
     [1.]
     [1.]
     [0.]]
                  precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00         2
             1.0       1.00      1.00      1.00         2
    
        accuracy                           1.00         4
       macro avg       1.00      1.00      1.00         4
    weighted avg       1.00      1.00      1.00         4
    
    
