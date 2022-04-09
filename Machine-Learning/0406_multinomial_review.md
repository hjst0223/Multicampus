# Logistic Regression 구현하기 😯
- 위스콘신 유방암 data set


```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Raw Data Set Loading
cancer = load_breast_cancer()

x_data = cancer.data    # 2차원 ndarray - 독립변수, feature
t_data = cancer.target  # 1차원 ndarray - 종속변수, label

train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(x_data,
                 t_data,
                 test_size=0.3,
                 stratify=t_data,
                 random_state=2)

# Model 생성
model = linear_model.LogisticRegression()

# Model 학습
model.fit(train_x_data, train_t_data)

# accuracy로 model 평가
test_score = model.score(test_x_data, test_t_data)

print('Logistic Regression Model의 정확도 : {}'.format(test_score))
```

    Logistic Regression Model의 정확도 : 0.9473684210526315
    

## SGD Classifier


```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Raw Data Set Loading
cancer = load_breast_cancer()

x_data = cancer.data    # 2차원 ndarray - 독립변수, feature
t_data = cancer.target  # 1차원 ndarray - 종속변수, label

train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(x_data,
                 t_data,
                 test_size=0.3,
                 stratify=t_data,
                 random_state=2)

# Model 생성
sgd = linear_model.SGDClassifier(loss='log',  # logistic regression 이용
                                 tol=1e-5,    # 얼마나 반복할건지를 loss값으로 설정 
                                 random_state=2)
# Model 학습
sgd.fit(train_x_data, train_t_data)

# Accuracy 측정
test_score = sgd.score(test_x_data, test_t_data)

print('SGDClassifier의 정확도 : {}'.format(test_score))
# 0.8947368421052632 정규화하지 않았으므로 각 feature마다 scale이 제각각임
```

    SGDClassifier의 정확도 : 0.8947368421052632
    

## 정규화를 이용한 SGD Classifier


```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Raw Data Set Loading
cancer = load_breast_cancer()

x_data = cancer.data    # 2차원 ndarray - 독립변수, feature
t_data = cancer.target  # 1차원 ndarray - 종속변수, label

train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(x_data,
                 t_data,
                 test_size=0.3,
                 stratify=t_data,
                 random_state=2)

# Data 정규화
scaler = StandardScaler()
scaler.fit(train_x_data)

# Model 생성
sgd = linear_model.SGDClassifier(loss='log',  # logistic regression 이용
                                 tol=1e-5,    # 얼마나 반복할건지를 loss값으로 설정 
                                 random_state=2)
# Model 학습
sgd.fit(scaler.transform(train_x_data), train_t_data)

# Accuracy 측정
test_score = sgd.score(scaler.transform(test_x_data), test_t_data)

print('정규화를 이용한 SGDClassifier의 정확도 : {}'.format(test_score))
```

    정규화를 이용한 SGDClassifier의 정확도 : 0.9649122807017544
    

## 정규화와 L2 Regularization을 이용한 SGD Classifier


```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Raw Data Set Loading
cancer = load_breast_cancer()

x_data = cancer.data    # 2차원 ndarray - 독립변수, feature
t_data = cancer.target  # 1차원 ndarray - 종속변수, label

train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(x_data,
                 t_data,
                 test_size=0.3,
                 stratify=t_data,
                 random_state=2)

# Data 정규화
scaler = StandardScaler()
scaler.fit(train_x_data)

# Model 생성
sgd = linear_model.SGDClassifier(loss='log',    # logistic regression 이용
                                 tol=1e-5,      # 얼마나 반복할건지를 loss값으로 설정 
                                 penalty='l2',  # L2 규제 이용 
                                 alpha=0.001,   # 규제 강도     
                                 random_state=2)
# Model 학습
sgd.fit(scaler.transform(train_x_data), train_t_data)

# Accuracy 측정
test_score = sgd.score(scaler.transform(test_x_data), test_t_data)

print('정규화와 규제를 이용한 SGDClassifier의 정확도 : {}'.format(test_score))
# 0.9649122807017544 => 0.9707602339181286 (규제를 이용하면 조금 더 나은 모델 만들 수 있음)
```

    정규화와 규제를 이용한 SGDClassifier의 정확도 : 0.9707602339181286
    

## sklearn으로 구현하고 성능평가 진행하기
- BMI 예제
- 성능평가의 metric - accuracy


```python
%reset

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats
import tensorflow as tf

# Raw Data Loading
df = pd.read_csv('./data/bmi.csv', skiprows=3)
display(df.head())
print(df.shape)
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>188</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>161</td>
      <td>68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>178</td>
      <td>52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>136</td>
      <td>63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>145</td>
      <td>52</td>
    </tr>
  </tbody>
</table>
</div>


    (20000, 3)
    

### 1. 데이터 전처리하기


```python
# 결측치 확인
print(df.isnull().sum())  # 결측치 없음
```

    label     0
    height    0
    weight    0
    dtype: int64
    


```python
# 이상치 확인 : z-score 방식
zscore_threshold = 2.0
print((np.abs(stats.zscore(df['height'])) > zscore_threshold).sum())  # => 0이면 이상치가 없음
print((np.abs(stats.zscore(df['weight'])) > zscore_threshold).sum())  # => 0이면 이상치가 없음
print(np.unique(df['label'], return_counts=True))  # 데이터 편향 존재 X
```

    0
    0
    (array([0, 1, 2], dtype=int64), array([6470, 5857, 7673], dtype=int64))
    

### 2. 정규화(Normalization) 하기


```python
# 먼저 train data와 validation data를 분리한 후 정규화 진행
train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(df[['height', 'weight']],
                 df['label'],
                 test_size=0.3,
                 random_state=1,
                 stratify=df['label'])

scaler = MinMaxScaler()
scaler.fit(train_x_data)

norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)
```

### 3. Model 생성 후 학습 및 평가


```python
model = linear_model.LogisticRegression(C=100000)
# 규제 적용(L2 규제)
# alpha값 정해야 함 
# 규제 강도 C = 1 / alpha

model.fit(norm_train_x_data, train_t_data)

# 평가를 위한 예측결과 얻기
predict_val = model.predict(norm_test_x_data)

# 나온 예측결과와 test_t_data 비교하기
acc = accuracy_score(predict_val, test_t_data)

print('sklearn으로 구한 Accuracy : {}'.format(acc))

# prediction
result = model.predict(scaler.transform(np.array([[187, 81]])))
print(result)  # 표준
```

    sklearn으로 구한 Accuracy : 0.9845
    [1]
    

## Tensorflow로 구현하고 성능평가 진행하기
- BMI 예제

### 1. batch 처리 X


```python
# multinomial 문제이므로 label(train_t_data, test_t_data)을 one-hot encoding 처리
# tensorflow의 기능을 이용해서 변경 => tensorflow node로 생성

sess = tf.Session()

onehot_train_t_data = sess.run(tf.one_hot(train_t_data, depth=3))  # depth는 class의 개수
onehot_test_t_data = sess.run(tf.one_hot(test_t_data, depth=3))    # depth는 class의 개수

# tensorflow graph 그리기
X = tf.placeholder(shape=[None,2], dtype=tf.float32)
T = tf.placeholder(shape=[None,3], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([2,3]))
b = tf.Variable(tf.random.normal([3]))

# model, Hypothesis
logit = tf.matmul(X,W) + b
H = tf.nn.softmax(logit)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                                 labels=T))
# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# session, 초기화
sess.run(tf.global_variables_initializer())

# 반복 학습
for step in range(10000):
    _, loss_val = sess.run([train, loss], feed_dict={X:norm_train_x_data,
                                                     T:onehot_train_t_data})                           
    if step % 1000 == 0:
        print('loss value : {}'.format(loss_val))
```

    WARNING:tensorflow:From C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_29596\2371570974.py:4: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    WARNING:tensorflow:From C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_29596\2371570974.py:10: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_29596\2371570974.py:25: The name tf.train.GradientDescentOptimizer is deprecated. Please use tf.compat.v1.train.GradientDescentOptimizer instead.
    
    WARNING:tensorflow:From C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_29596\2371570974.py:28: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    loss value : 1.1495620012283325
    loss value : 0.5054112076759338
    loss value : 0.40890854597091675
    loss value : 0.35897088050842285
    loss value : 0.32620033621788025
    loss value : 0.3023587465286255
    loss value : 0.28392332792282104
    loss value : 0.2690742611885071
    loss value : 0.2567571699619293
    loss value : 0.24631084501743317
    

### 2. batch 처리 O
- 반복 학습 시 주의점: 학습데이터의 사이즈가 매우 크면 메모리에 데이터를 한번에 모두 loading할 수 없음 
- -> memory fault나면서 수행 중지됨
- => batch처리로 해결


```python
# multinomial 문제이므로 label(train_t_data, test_t_data)을 one-hot encoding 처리
# tensorflow의 기능을 이용해서 변경 => tensorflow node로 생성
sess = tf.Session()

onehot_train_t_data = sess.run(tf.one_hot(train_t_data, depth=3))  # depth는 class의 개수
onehot_test_t_data = sess.run(tf.one_hot(test_t_data, depth=3))    # depth는 class의 개수

# tensorflow graph 그리기
X = tf.placeholder(shape=[None,2], dtype=tf.float32)
T = tf.placeholder(shape=[None,3], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([2,3]))
b = tf.Variable(tf.random.normal([3]))

# model, Hypothesis
logit = tf.matmul(X,W) + b
H = tf.nn.softmax(logit)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                                 labels=T))
# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# session, 초기화
sess.run(tf.global_variables_initializer())

# 반복 학습
num_of_epoch = 1000  # 학습을 위한 전체 epoch 수
num_of_batch = 100   # 한번에 학습할 데이터 량

for step in range(num_of_epoch):
    total_batch = int(norm_train_x_data.shape[0] / num_of_batch)
    
    for i in range(total_batch):
        batch_x = norm_train_x_data[i*num_of_batch:(i+1)*num_of_batch]
        batch_y = onehot_train_t_data[i*num_of_batch:(i+1)*num_of_batch]
        _, loss_val = sess.run([train, loss], 
                               feed_dict={X:batch_x,
                                          T:batch_y})                           
    if step % 100 == 0:
        print('loss value : {}'.format(loss_val))
```

    loss value : 0.7061227560043335
    loss value : 0.1640716791152954
    loss value : 0.12247326225042343
    loss value : 0.10330770164728165
    loss value : 0.09181509166955948
    loss value : 0.08399397879838943
    loss value : 0.0782523825764656
    loss value : 0.07381793856620789
    loss value : 0.07026541233062744
    loss value : 0.06734078377485275
    


```python
# 학습 종료 후 성능평가(Accuracy)
result = sess.run(H, feed_dict={X:scaler.transform(np.array([[187,81]]))})
print(result)                     # 값이 가장 큰 것이 이 데이터의 class => 1
print(np.argmax(result, axis=1))  # 가장 큰 값의 index 알려줌
```

    [[4.7410915e-05 9.0800029e-01 9.1952242e-02]]
    [1]
    


```python
predict = tf.argmax(H,1)
correct = tf.equal(predict, tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

result = sess.run(accuracy, feed_dict={X:norm_test_x_data,
                                       T:onehot_test_t_data})
print(result)
```

    0.9855
    
