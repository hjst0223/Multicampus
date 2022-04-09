# Tensorflow 1.15 버전으로 MNIST 예제 구현하기
- Data : Kaggle에서 다운로드


```python
import tensorflow as tf

print(tf.__version__)  
```

    1.15.0
    


```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Raw Data Loading
df = pd.read_csv('./data/kaggle/mnist/train.csv')
display(df.shape)
```


    (42000, 785)


- 데이터 전처리 - 결측치, 이상치 존재 X, 정규화 필요

## 이미지 확인하기


```python
figure = plt.figure()
ax_arr = []  # python list

img_data = df.drop('label', axis=1, inplace=False).values

for n in range(10):
    ax_arr.append(figure.add_subplot(2,5,n+1))
    ax_arr[n].imshow(img_data[n].reshape(28,28), 
                     cmap='Greys',            # 흑백이미지 표현
                     interpolation='nearest') # 보간법

plt.tight_layout()
plt.show()
```


    
![png](/Machine-Learning/images/0407/output_6_0.png)
    



```python
# Data Split
train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(df.drop('label', axis=1, inplace=False),
                 df['label'],
                 test_size=0.3,
                 random_state=1,
                 stratify=df['label'])
# 정규화
scaler = MinMaxScaler()
scaler.fit(train_x_data)

norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)
```


```python
## Tensorflow Implementation ##
sess = tf.Session()

onehot_train_t_data = sess.run(tf.one_hot(train_t_data, depth=10))
onehot_test_t_data = sess.run(tf.one_hot(test_t_data, depth=10))

# placeholder
X = tf.placeholder(shape=[None,784], dtype=tf.float32)
T = tf.placeholder(shape=[None,10], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([784,10]))
b = tf.Variable(tf.random.normal([10]))

# Hypothesis, Model
logit = tf.matmul(X,W) + b
H = tf.nn.softmax(logit)

# Loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                                 labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# session, 초기화
sess.run(tf.global_variables_initializer())

# 반복학습
num_of_epoch = 1000
batch_size = 100

for step in range(num_of_epoch):
    
    total_batch = int(norm_train_x_data.shape[0] / batch_size)

    for i in range(total_batch):
        batch_x = norm_train_x_data[i*batch_size:(i+1)*batch_size]
        batch_y = onehot_train_t_data[i*batch_size:(i+1)*batch_size]
        
        _, loss_val = sess.run([train, loss], feed_dict={X:batch_x,
                                                         T:batch_y})
    if step % 100 == 0:
        print('loss val : {}'.format(loss_val))
```

    
    loss val : 1.3088774681091309
    loss val : 0.2624996304512024
    loss val : 0.2428320348262787
    loss val : 0.2362198680639267
    loss val : 0.2295105755329132
    loss val : 0.22287388145923615
    loss val : 0.21694685518741608
    loss val : 0.21189232170581818
    loss val : 0.20764656364917755
    loss val : 0.2040848582983017
    


```python
# accuracy 측정

predict = tf.argmax(H,1)
correct = tf.equal(predict, tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

accuracy_val = sess.run(accuracy, feed_dict={X:norm_test_x_data,
                                             T:onehot_test_t_data})
print('Accuracy : {}'.format(accuracy_val))
```

    Accuracy : 0.908650815486908
    
