# Logistic Regression 구현하기 🙄


```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import linear_model  # sklearn으로 logistic 구현
from sklearn.preprocessing import MinMaxScaler  # 정규화 진행
from scipy import stats  # 이상치 처리
import matplotlib.pyplot as plt
import warnings

# 경고메시지 출력 안 되게 설정
warnings.filterwarnings(action='ignore')

# Raw Data Loading
df = pd.read_csv('./data/admission.csv')
```

## Preprocessing

### 1. 결측치 처리하기


```python
# df의 sum() : axis 생략 시 (axis=1) column당 합 
# numpy의 sum() : axis 생략 시 전체 합
print(df.isnull().sum())   # 결치값 없음
df.info()
```

    admit    0
    gre      0
    gpa      0
    rank     0
    dtype: int64
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 400 entries, 0 to 399
    Data columns (total 4 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   admit   400 non-null    int64  
     1   gre     400 non-null    int64  
     2   gpa     400 non-null    float64
     3   rank    400 non-null    int64  
    dtypes: float64(1), int64(3)
    memory usage: 12.6 KB
    

### 2. 이상치 처리하기
- 종속변수의 이상치 : outlier
- 독립변수의 이상치 : 지대값
- 대부분의 경우 outlier라 함
- boxplot

#### boxplot으로 확인하기


```python
figure = plt.figure()
ax1 = figure.add_subplot(1,4,1)
ax2 = figure.add_subplot(1,4,2)
ax3 = figure.add_subplot(1,4,3)
ax4 = figure.add_subplot(1,4,4)
ax1.set_title('ADMIT')
ax2.set_title('GRE')
ax3.set_title('GPA')
ax4.set_title('RANK')

ax1.boxplot(df['admit'])
ax2.boxplot(df['gre'])
ax3.boxplot(df['gpa'])
ax4.boxplot(df['rank'])

# 레이아웃 조절
figure.tight_layout()

plt.show()  # 이상치 존재
```


    
![png](/Machine-Learning/images/0404/output_7_0.png)
    


#### z-score로 이상치 제거하기


```python
zscore_threshold = 2.0

for col in df.columns:
    outlier = df[col][np.abs(stats.zscore(df[col])) > zscore_threshold]
    df = df.loc[~df[col].isin(outlier)]
    
display(df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>admit</th>
      <th>gre</th>
      <th>gpa</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>380</td>
      <td>3.61</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>660</td>
      <td>3.67</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>800</td>
      <td>4.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>640</td>
      <td>3.19</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>520</td>
      <td>2.93</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>394</th>
      <td>1</td>
      <td>460</td>
      <td>3.99</td>
      <td>3</td>
    </tr>
    <tr>
      <th>395</th>
      <td>0</td>
      <td>620</td>
      <td>4.00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>396</th>
      <td>0</td>
      <td>560</td>
      <td>3.04</td>
      <td>3</td>
    </tr>
    <tr>
      <th>398</th>
      <td>0</td>
      <td>700</td>
      <td>3.65</td>
      <td>2</td>
    </tr>
    <tr>
      <th>399</th>
      <td>0</td>
      <td>600</td>
      <td>3.89</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>382 rows × 4 columns</p>
</div>


### 3. 정규화 작업하기


```python
x_data = df.drop('admit', axis=1, inplace=False)
t_data = df['admit'].values.reshape(-1,1)  # t_data는 0과 1로만 구성되어 있기 때문에 정규화 필요 X

# 정규화를 위한 scaler
scaler = MinMaxScaler()
scaler.fit(x_data)

norm_x_data = scaler.transform(x_data)
print(norm_x_data)
```

    [[0.04545455 0.71111111 0.66666667]
     [0.68181818 0.75555556 0.66666667]
     [1.         1.         0.        ]
     ...
     [0.45454545 0.28888889 0.66666667]
     [0.77272727 0.74074074 0.33333333]
     [0.54545455 0.91851852 0.66666667]]
    

#### training data set
- norm_x_data
- t_data

#### sklearn 구현


```python
model = linear_model.LogisticRegression()

model.fit(x_data, t_data)

my_score = np.array([[600, 3.8, 1]])
predict_val = model.predict(my_score)          # 0 or 1로 결과 도출
predict_proba = model.predict_proba(my_score)  # 확률값으로 결과 도출

print('sklearn의 결과 : 합격여부 : {}, 확률 : {}'.format(predict_val, predict_proba))
```

    sklearn의 결과 : 합격여부 : [1], 확률 : [[0.43740782 0.56259218]]
    

#### tensorflow 구현


```python
# placeholder
X = tf.placeholder(shape=[None,3], dtype=tf.float32)
T = tf.placeholder(shape=[None,1], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([3,1]))
b = tf.Variable(tf.random.normal([1]))

# Hypothesis, model, predict model, logistic regression model
logit = tf.matmul(X,W) + b  # linear regression
H = tf.sigmoid(logit)

# loss function, cross entropy, log loss
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

# Session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 반복학습
for step in range(300000):
    _, loss_val = sess.run([train, loss], 
                           feed_dict={X: norm_x_data, T: t_data})
    
    if step % 30000 == 0:
        print('loss의 값 : {}'.format(loss_val))
```


    loss의 값 : 1.1713826656341553
    loss의 값 : 0.6711441874504089
    loss의 값 : 0.632067084312439
    loss의 값 : 0.6239297389984131
    loss의 값 : 0.6186142563819885
    loss의 값 : 0.6141023635864258
    loss의 값 : 0.6101633310317993
    loss의 값 : 0.606709361076355
    loss의 값 : 0.6036775708198547
    loss의 값 : 0.6010180711746216
    


```python
# predict
my_score = np.array([[600, 3.8, 1]])
norm_my_score = scaler.transform(my_score)

result = sess.run(H, feed_dict={X: norm_my_score})
print('tensorflow로 예측한 결과 : {}'.format(result))
```

    tensorflow로 예측한 결과 : [[0.4217303]]
    

# Regression의 Metrics 😉


```python
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy import stats
from sklearn.model_selection import train_test_split


### Raw Data Loading
df = pd.read_csv('./data/ozone.csv')
# print(df.shape)  # (153, 6)

# 결측치 처리
training_data = df.dropna(how='any', inplace=False)
# print(training_data.shape)  # (111, 6)

# 이상치 처리
zscore_threshold = 2.0

for col in training_data.columns:
    outlier = training_data[col][np.abs(stats.zscore(training_data[col])) > zscore_threshold]
    training_data = training_data.loc[~training_data[col].isin(outlier)]
    
# sklearn => 정규화 처리할 필요 없음
# display(training_data.head())

# Data Set
x_data = training_data[['Solar.R', 'Wind', 'Temp']].values
t_data = training_data['Ozone'].values.reshape(-1,1)

# Train / Validation Data Set
train_x_data, valid_x_data, train_t_data, valid_t_data = \
train_test_split(x_data,
                 t_data,
                 test_size=0.3,
                 random_state=2)  # random의 seed 역할

# Model
model = linear_model.LinearRegression()

# Model 학습
model.fit(train_x_data, train_t_data)

# 예측값(predict_value)
# 정답(valid_t_data)
predict_value = model.predict(valid_x_data)

# 예측값과 정답간의 차이가 작을수록 좋음
```

## 1. MAE(Mean Absolute Error) 
- 예측값과 정답 간 차의 절대값의 평균
- 직관적, 단위 같음
- scale에 따라 의존적


```python
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(valid_t_data, predict_value))
```

    13.924465776324642
    

## 2. MSE(Mean Squard Error)
- 예측값과 정답 간 차의 제곱의 평균
- MAE보다 error에 가중치(제곱)을 주는 Metrics
- MAE보다 이상치에 더 민감함


```python
from sklearn.metrics import mean_squared_error

print(mean_squared_error(valid_t_data, predict_value)) 
```

    271.56711923670605
    

## 3. R squared 
- 분산을 기반으로 한 평가지표
- (예측값의 variance) / (정답의 variance)
- 1과 가까울수록 좋은 모델이라 할 수 있음


```python
from sklearn.metrics import r2_score

print(r2_score(valid_t_data, predict_value)) 
```

    0.3734728354920863
    
