# CNN 구현하기 🍑

## 1. channel이 3인 이미지 데이터로 convolution 연산하기

- 입력데이터의 형태
- (이미지의 개수, 이미지의 height, 이미지의 width, channel)
- (1, 3, 3, 3)


```python
import numpy as np
import tensorflow as tf

image = np.array([[[[1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3]],
                   [[1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3]],
                   [[1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3]]]], dtype=np.float64)

print(image.shape)
```

    (1, 3, 3, 3)
    

- filter의 형태
- (filter의 height, filter의 width, filter의 channel, filter의 개수)
- (2, 2, 3, 2)


```python
weight = np.array([[[[1, 2],
                     [1, 2],
                     [1, 2]],
                    [[1, 2],
                     [1, 2],
                     [1, 2]]],
                   [[[1, 2],
                     [1, 2],
                     [1, 2]],
                    [[1, 2],
                     [1, 2],
                     [1, 2]]]], dtype=np.float64)

print(weight.shape)
```

    (2, 2, 3, 2)
    


```python
conv2d = tf.nn.conv2d(image,
                      weight,
                      strides=[1, 1, 1, 1],
                      padding='VALID')

sess = tf.Session()
result = sess.run(conv2d)
print(result)
```


    [[[[24. 48.]
       [24. 48.]]
    
      [[24. 48.]
       [24. 48.]]]]
    

## 2. channel이 3인 이미지 데이터로 pooling 연산하기

- 입력데이터의 형태
- (이미지의 개수, 이미지의 height, 이미지의 width, channel)
- (1, 429, 640, 3)


```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img

ori_image = img.imread('./images/girl-teddy.jpg')  # 원본 이미지
print(type(ori_image))
print(ori_image.shape)  # (세로, 가로, channel) 3차원
```

    <class 'numpy.ndarray'>
    (429, 640, 3)
    


```python
figure = plt.figure()

ax1 = figure.add_subplot(1, 3, 1)
ax2 = figure.add_subplot(1, 3, 2)
ax3 = figure.add_subplot(1, 3, 3)

ax1.imshow(ori_image)

input_image = ori_image.reshape((1, ) + ori_image.shape)  # 3차원 => 4차원
print(input_image.shape)  # 4차원
input_image = input_image.astype(np.float32)  # 데이터를 실수로 변환

# 이미지의 channel을 1로 변경
channel_1_input_image = input_image[:, :, :, 0:1]  # 4차원 유지하면서 값 1개만 가져옴
print(channel_1_input_image.shape)

# filter
# (3, 3, 1, 1)
weight = np.array([[[[-1]], [[0]], [[1]]],
                   [[[-1]], [[0]], [[1]]],
                   [[[-1]], [[0]], [[1]]]])
print(weight.shape)

# stride = 1
# padding = 'VALID'

conv2d = tf.nn.conv2d(channel_1_input_image,
                      weight,
                      strides=[1, 1, 1, 1],
                      padding='VALID')

sess = tf.Session()
result = sess.run(conv2d)
print(result.shape)  # (이미지 개수, 특징을 뽑아낸 이미지의 height, width, filter 개수)

t_img = result[0, :, :, :]  # 4차원 => 3차원
print(t_img.shape)
ax2.imshow(t_img)

# pooling 처리
pooling_result = tf.nn.max_pool(result,
                                ksize=[1, 3, 3, 1],    # 3 * 3 kernel
                                strides=[1, 3, 3, 1],  # kernel size와 동일하게 설정)
                                padding='VALID')

pool_img = sess.run(pooling_result)
pool_img = pool_img[0, :, :, :]  # 4차원 => 3차원
print(pool_img.shape)
ax3.imshow(pool_img)

plt.tight_layout()
plt.show()
```

    (1, 429, 640, 3)
    (1, 429, 640, 1)
    (3, 3, 1, 1)
    (1, 427, 638, 1)
    (427, 638, 1)
    (142, 212, 1)
    


    
![png](/Machine-Learning/images/0414/output_10_1.png)
    


## MNIST 예제로 학습시킬 이미지 만들기


```python
%reset
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y
    


```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img

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



```python
# subplot 생성
figure = plt.figure()
ax = list()

for i in range(5):
    ax.append(figure.add_subplot(1, 5, i + 1))
    
# 이미지 확인
img_data = df.drop('label', axis=1, inplace=False).values

ori_image = img_data[5:6].reshape(28, 28)  # 2차원 이미지 형태
ax[0].imshow(ori_image, cmap='Greys')

# 입력 이미지
# (1, 28, 28, 1) => (이미지 개수, height, width, channel)
input_image = ori_image.reshape((1, ) + ori_image.shape + (1, ))  # 2차원 => 4차원
print(input_image.shape)
input_image = input_image.astype(np.float32)  # 데이터를 실수로 변환

# filter 
# (3, 3, 1, 4) => (height, width, channel, filter 개수)
# rand() => 균등분포로 0과 1 사이의 난수 발생
weight = np.random.rand(3, 3, 1, 4)
print(weight.shape)

conv2d = tf.nn.conv2d(input_image,
                      weight,
                      strides=[1, 1, 1, 1],
                      padding='VALID')

sess = tf.Session()
conv2d_result = sess.run(conv2d)

relu_ = tf.nn.relu(conv2d_result)
relu_result = sess.run(relu_)  # relu 처리 결과

pool = tf.nn.max_pool(relu_result,
                      ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1],
                      padding='VALID')

pool_result = sess.run(pool)

print(pool_result.shape)

# 1과 4의 위치를 swap
i = np.swapaxes(pool_result, 0, 3)
print(i.shape)

# (4, 13, 13, 1)
# ↓     ↓
# idx   image(3차원)
for idx, image in enumerate(i):  
    ax[idx + 1].imshow(image.squeeze(), cmap='Greys')
    # squeeze()는 numpy에서 차원이 1인 것 제거  
    
plt.tight_layout()
plt.show()
```

    (1, 28, 28, 1)
    (3, 3, 1, 4)
    (1, 13, 13, 4)
    (4, 13, 13, 1)
    


    
![png](/Machine-Learning/images/0414/output_14_1.png)
    



```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

tf.reset_default_graph()  # tensorflow graph 리셋

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


- 결측치와 이상치 없음


```python
# Data Split
train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(df.drop('label', axis=1, inplace=False),
                 df['label'],
                 test_size=0.3,
                 random_state=1,
                 stratify=df['label'])

# x_data에 대해서는 정규화 진행
scaler = MinMaxScaler()
scaler.fit(train_x_data)
norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)

# t_data에 대해서는 one-hot encoding 처리
sess = tf.Session()
onehot_train_t_data = sess.run(tf.one_hot(train_t_data, depth=10))
onehot_test_t_data = sess.run(tf.one_hot(test_t_data, depth=10))
```

### Tensorflow 구현


```python
# placeholder
X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
T = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# 입력 x_data의 shape 변경 (2차원 => 4차원)
x_img = tf.reshape(X, [-1, 28, 28, 1])  # (이미지 개수, height, width, channel)

# convolution layer 처리
# filter의 shape => (height, width, channel, filter 개수)
W2 = tf.Variable(tf.random.normal([3, 3, 1, 32]))
L1 = tf.nn.conv2d(x_img, W2, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
print(f'L1의 결과 데이터 shape : {L1.shape}')

L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(f'L1의 pooling 결과 데이터 shape : {L1.shape}')
```


    L1의 결과 데이터 shape : (?, 28, 28, 32)
    L1의 pooling 결과 데이터 shape : (?, 14, 14, 32)
    


```python
# 2번째 convolution layer
W3 = tf.Variable(tf.random.normal([3, 3, 32, 64]))  # filter의 개수(64) 정하기
L2 = tf.nn.conv2d(L1, W3, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
print(f'L2의 결과 데이터 shape : {L2.shape}')

L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(f'L2의 pooling 결과 데이터 shape : {L2.shape}')
```

    L2의 결과 데이터 shape : (?, 14, 14, 64)
    L2의 pooling 결과 데이터 shape : (?, 7, 7, 64)
    


```python
# 이렇게 나온 데이터를 DNN에 넣어서 학습
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])

# Weight & bias
# W4 : 들어오는 feature들에 대한 가중치
W4 = tf.get_variable('W4', shape=[7 * 7 * 64, 256],
                     initializer=tf.contrib.layers.variance_scaling_initializer())  # he's initailization
b4 = tf.Variable(tf.random.normal([256]))

_layer3 = tf.matmul(L2, W4) + b4
layer3 = tf.nn.relu(_layer3)
layer3 = tf.nn.dropout(layer3, rate=0.3)

W5 = tf.get_variable('W5', shape=[256, 10],
                     initializer=tf.contrib.layers.variance_scaling_initializer())  # he's initailization
b5 = tf.Variable(tf.random.normal([10]))

# Hypothesis, model
logit = tf.matmul(layer3, W5) + b5
H = tf.nn.softmax(logit)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                                 labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

# session 및 초기화
sess.run(tf.global_variables_initializer())

# 반복학습 (원래는 batch 처리해서 학습해야 함)
for step in range(200):
    
    tmp, loss_val = sess.run([train, loss],
                             feed_dict={X:norm_train_x_data,
                                        T:onehot_train_t_data})
    
    if step % 20 == 0:
        print(f'loss value : {loss_val}')
```
    

    loss value : 35.11565399169922
    loss value : 1.5435703992843628
    loss value : 1.0705573558807373
    loss value : 0.899503231048584
    loss value : 0.8002464175224304
    loss value : 0.730694591999054
    loss value : 0.6877204179763794
    loss value : 0.6340042352676392
    loss value : 0.5964807868003845
    loss value : 0.5663800239562988
    
