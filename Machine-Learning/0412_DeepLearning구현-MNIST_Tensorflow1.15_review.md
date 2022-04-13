# MNIST 예제로 Deep Learning 구현하기 😳


```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Raw Data Loading
df = pd.read_csv('./data/kaggle/mnist/train.csv')
display(df.head())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>


## 이미지 확인하기


```python
# 결측치와 이상치 존재 X
img_data = df.drop('label', axis=1, inplace=False).values

figure = plt.figure()
ax_arr = []

for n in range(10):
    ax_arr.append(figure.add_subplot(2, 5, n + 1))
    ax_arr[n].imshow(img_data[n].reshape(28, 28),
                     cmap='Greys',             # 흑백
                     interpolation='nearest')  # 이미지가 깨지는 것 보정

plt.tight_layout()
plt.show()
```


    
![png](/Machine-Learning/images/0412/output_4_0.png)
    


## 데이터 분할하기
- train 데이터와 validation 데이터


```python
train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(df.drop('label', axis=1, inplace=False),
                 df['label'],
                 test_size=0.3,
                 random_state=1,
                 stratify=df['label'])
```

## 정규화하기
- train_x_data, test_x_data에만


```python
scaler = MinMaxScaler()
scaler.fit(train_x_data)

norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)
```

## one-hot encoding 처리하기
- train_t_data, test_t_data


```python
sess = tf.Session()

onehot_train_t_data = sess.run(tf.one_hot(train_t_data, depth=10))
onehot_test_t_data = sess.run(tf.one_hot(test_t_data, depth=10))
```

    WARNING:tensorflow:From C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_17156\1037498019.py:1: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    

## Tensorflow 1.15로 구현하기
### 1. 일반 multinomial classification 구현하기


```python
# placeholder
X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
T = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([784, 10]))
b = tf.Variable(tf.random.normal([10]))

# Hypothesis, model
logit = tf.matmul(X, W) + b
H = tf.nn.softmax(logit)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                                 labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# 초기화
sess.run(tf.global_variables_initializer())

# 반복 학습
for step in range(1000):
    
    tmp, loss_val = sess.run([train, loss],
                             feed_dict={X:norm_train_x_data,
                                        T:onehot_train_t_data})
    if step % 100 == 0:
        print(f'loss 값 : {loss_val}')
```

    WARNING:tensorflow:From C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_17156\2745053351.py:2: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_17156\2745053351.py:18: The name tf.train.GradientDescentOptimizer is deprecated. Please use tf.compat.v1.train.GradientDescentOptimizer instead.
    
    WARNING:tensorflow:From C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_17156\2745053351.py:21: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    loss 값 : 18.254226684570312
    loss 값 : 4.129348278045654
    loss 값 : 2.5434889793395996
    loss 값 : 1.926551103591919
    loss 값 : 1.6057692766189575
    loss 값 : 1.4100968837738037
    loss 값 : 1.2774170637130737
    loss 값 : 1.1804654598236084
    loss 값 : 1.1057480573654175
    loss 값 : 1.0458576679229736
    

#### 성능 평가 - Accuracy


```python
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(T, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

train_result = sess.run(accuracy,
                        feed_dict={X:norm_train_x_data,
                                   T:onehot_train_t_data})
print(f'train 데이터의 정확도 : {train_result}')

result = sess.run(accuracy,
                  feed_dict={X:norm_test_x_data,
                             T:onehot_test_t_data})
print(f'test 데이터의 정확도 : {result}')
```

    train 데이터의 정확도 : 0.796700656414032
    test 데이터의 정확도 : 0.7917460203170776
    

=> underfitting!
- epochs 조정 (1000= > 5000)


```python
# placeholder
X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
T = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([784, 10]))
b = tf.Variable(tf.random.normal([10]))

# Hypothesis, model
logit = tf.matmul(X, W) + b
H = tf.nn.softmax(logit)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                                 labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# 초기화
sess.run(tf.global_variables_initializer())

# 반복 학습
for step in range(5000):
    
    tmp, loss_val = sess.run([train, loss],
                             feed_dict={X:norm_train_x_data,
                                        T:onehot_train_t_data})
    if step % 500 == 0:
        print(f'loss 값 : {loss_val}')
```

    loss 값 : 11.037571907043457
    loss 값 : 1.3349248170852661
    loss 값 : 0.9527879357337952
    loss 값 : 0.7982969284057617
    loss 값 : 0.7100622057914734
    loss 값 : 0.651174008846283
    loss 값 : 0.608055055141449
    loss 값 : 0.5745164752006531
    loss 값 : 0.5473520159721375
    loss 값 : 0.5247088670730591
    

#### 성능 평가 - accuracy


```python
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(T, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

train_result = sess.run(accuracy,
                        feed_dict={X:norm_train_x_data,
                                   T:onehot_train_t_data})
print(f'train 데이터의 정확도 : {train_result}')

result = sess.run(accuracy,
                  feed_dict={X:norm_test_x_data,
                             T:onehot_test_t_data})
print(f'test 데이터의 정확도 : {result}')
```

    train 데이터의 정확도 : 0.8829591870307922
    test 데이터의 정확도 : 0.8741269707679749
    

### 2. Neural Network로 multinomial classification 구현하기
####  weight 초기화 / activation 변경 / dropout 제외


```python
# placeholder
X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
T = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# Weight & bias
W2 = tf.Variable(tf.random.normal([784, 256]))
b2 = tf.Variable(tf.random.normal([256]))
layer2 = tf.sigmoid(tf.matmul(X, W2) + b2)

W3 = tf.Variable(tf.random.normal([256, 128]))
b3 = tf.Variable(tf.random.normal([128]))
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random.normal([128, 10]))
b4 = tf.Variable(tf.random.normal([10]))

# Hypothesis, model
logit = tf.matmul(layer3, W4) + b4
H = tf.nn.softmax(logit)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                                 labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# 초기화
sess.run(tf.global_variables_initializer())

# 반복 학습
for step in range(1000):
    
    tmp, loss_val = sess.run([train, loss],
                             feed_dict={X:norm_train_x_data,
                                        T:onehot_train_t_data})
    if step % 100 == 0:
        print(f'loss 값 : {loss_val}')
```

    loss 값 : 12.547239303588867
    loss 값 : 2.1144533157348633
    loss 값 : 1.4673641920089722
    loss 값 : 1.1843092441558838
    loss 값 : 1.0173969268798828
    loss 값 : 0.9050785899162292
    loss 값 : 0.823430597782135
    loss 값 : 0.7608280181884766
    loss 값 : 0.7108744978904724
    loss 값 : 0.6697903871536255
    

#### 성능 평가 - accuracy
- H => (0.1 0.2 0.1 0.3 ... 0.1) => 3


```python
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(T, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

train_result = sess.run(accuracy,
                        feed_dict={X:norm_train_x_data,
                                   T:onehot_train_t_data})
print(f'train 데이터의 정확도 : {train_result}')

result = sess.run(accuracy,
                  feed_dict={X:norm_test_x_data,
                             T:onehot_test_t_data})
print(f'test 데이터의 정확도 : {result}')
```

    train 데이터의 정확도 : 0.8063605427742004
    test 데이터의 정확도 : 0.7842063307762146
    

### 3. Deep Learning으로 multinomial classification 구현하기
#### weight 초기화 / activation 변경 / dropout 포함
- dropout : overfitting을 피하기 위해서 전체 노드를 사용하지 않고 일부 노드만 사용하는 방법


```python
# placeholder
X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
T = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# Weight & bias
# W2 = tf.Variable(tf.random.normal([784, 256]))
W2 = tf.get_variable('W2', shape=[784, 256],
                     initializer=tf.contrib.layers.variance_scaling_initializer())
b2 = tf.Variable(tf.random.normal([256]))
_layer2 = tf.nn.relu(tf.matmul(X, W2) + b2)
layer2 = tf.nn.dropout(_layer2, rate=0.3)

# W3 = tf.Variable(tf.random.normal([256, 128]))
W3 = tf.get_variable('W3', shape=[256, 128],
                     initializer=tf.contrib.layers.variance_scaling_initializer())
b3 = tf.Variable(tf.random.normal([128]))
_layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
layer3 = tf.nn.dropout(_layer3, rate=0.3)

# W4 = tf.Variable(tf.random.normal([128, 10]))
W4 = tf.get_variable('W4', shape=[128, 10],
                     initializer=tf.contrib.layers.variance_scaling_initializer())
b4 = tf.Variable(tf.random.normal([10]))

# Hypothesis, model
logit = tf.matmul(layer3, W4) + b4
H = tf.nn.softmax(logit)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                                 labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# 초기화
sess.run(tf.global_variables_initializer())

# 반복 학습
for step in range(5000):
    
    tmp, loss_val = sess.run([train, loss],
                             feed_dict={X:norm_train_x_data,
                                        T:onehot_train_t_data})
    if step % 500 == 0:
        print(f'loss 값 : {loss_val}')
```

    WARNING:tensorflow:From C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_17156\2327548789.py:7: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    loss 값 : 5.908406734466553
    loss 값 : 0.34757450222969055
    loss 값 : 0.24586942791938782
    loss 값 : 0.19825856387615204
    loss 값 : 0.16740751266479492
    loss 값 : 0.14313630759716034
    loss 값 : 0.1306665688753128
    loss 값 : 0.12048626691102982
    loss 값 : 0.10242922604084015
    loss 값 : 0.093268483877182
    

#### 성능 평가 - accuracy
- H => (0.1 0.2 0.1 0.3 ... 0.1) => 3


```python
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(T, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

train_result = sess.run(accuracy,
                        feed_dict={X:norm_train_x_data,
                                   T:onehot_train_t_data})
print(f'train 데이터의 정확도 : {train_result}')

result = sess.run(accuracy,
                  feed_dict={X:norm_test_x_data,
                             T:onehot_test_t_data})
print(f'test 데이터의 정확도 : {result}')
```

    train 데이터의 정확도 : 0.9722108840942383
    test 데이터의 정확도 : 0.9558730125427246
    
