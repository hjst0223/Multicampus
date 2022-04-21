# Google Colaboratory 😮
### 현재 설치되어 있는 tensorflow의 버전이 2.x 버전이므로

### 현재 2.x버전 삭제 후 1.15버전 설치
```python
!pip uninstall tensorflow
```

```python
!pip install tensorflow==1.15
```

```python
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats
import tensorflow as tf

# Raw Data Loading
df = pd.read_csv('/content/drive/MyDrive/colab/bmi.csv', skiprows=3)
display(df.head())
print(df.shape)
```

<pre>
   label  height  weight
0      1     188      71
1      2     161      68
2      0     178      52
3      2     136      63
4      1     145      52
</pre>
<pre>
(20000, 3)
</pre>

```python
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


```python
sess = tf.Session()

onehot_train_t_data = sess.run(tf.one_hot(train_t_data, depth=3))  # depth는 class의 개수
onehot_test_t_data = sess.run(tf.one_hot(test_t_data, depth=3))

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

# 반복 학습 - batch 처리
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

<pre>
loss value : 0.6971952319145203
loss value : 0.16385383903980255
loss value : 0.12239065021276474
loss value : 0.10326226055622101
loss value : 0.09178560227155685
loss value : 0.08397280424833298
loss value : 0.07823646813631058
loss value : 0.07380534708499908
loss value : 0.07025516033172607
loss value : 0.06733208894729614
</pre>

```python
predict = tf.argmax(H,1)
correct = tf.equal(predict, tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

result = sess.run(accuracy, feed_dict={X:norm_test_x_data,
                                       T:onehot_test_t_data})
print(result)
```

<pre>
0.9855
</pre>
