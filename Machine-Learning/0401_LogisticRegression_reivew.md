# Logistic Regression을 Graphic하게 알아보기 🧐


```python
import numpy as np
from sklearn import linear_model
import mglearn   # utility module
import matplotlib.pyplot as plt
import warnings  # warning 출력되지 않게

warnings.filterwarnings(action='ignore')  # 경고 메시지 출력하지 않음

# Training Data Set - x는 독립변수, y는 종속변수 
x, y = mglearn.datasets.make_forge()  # make_forge(): 특정 dataset이 주어짐
```


```python
print(x)  # 독립변수(2차원 ndarray)
```

    [[ 9.96346605  4.59676542]
     [11.0329545  -0.16816717]
     [11.54155807  5.21116083]
     [ 8.69289001  1.54322016]
     [ 8.1062269   4.28695977]
     [ 8.30988863  4.80623966]
     [11.93027136  4.64866327]
     [ 9.67284681 -0.20283165]
     [ 8.34810316  5.13415623]
     [ 8.67494727  4.47573059]
     [ 9.17748385  5.09283177]
     [10.24028948  2.45544401]
     [ 8.68937095  1.48709629]
     [ 8.92229526 -0.63993225]
     [ 9.49123469  4.33224792]
     [ 9.25694192  5.13284858]
     [ 7.99815287  4.8525051 ]
     [ 8.18378052  1.29564214]
     [ 8.7337095   2.49162431]
     [ 9.32298256  5.09840649]
     [10.06393839  0.99078055]
     [ 9.50048972 -0.26430318]
     [ 8.34468785  1.63824349]
     [ 9.50169345  1.93824624]
     [ 9.15072323  5.49832246]
     [11.563957    1.3389402 ]]
    


```python
print(y)  # 종속변수(1차원 ndarray) 
```

    [1 0 1 0 0 1 1 0 1 1 1 1 0 0 1 1 1 0 0 1 0 0 0 0 1 0]
    


```python
mglearn.discrete_scatter(x[:,0], x[:,1], y)
plt.legend(['class 0', 'class 1'], loc='best')  # 범례
plt.show()
```


    
![png](/Machine-Learning/images/0401/output_4_0.png)
    


## 경계선을 구하기 위해 linear regression을 구한다.
## => 이 경계선을 기준으로 어느 쪽에 속해있는지 찾는 것이 logistic regression


```python
mglearn.discrete_scatter(x[:,0], x[:,1], y)
plt.legend(['class 0', 'class 1'], loc='best')  # 범례

model = linear_model.LinearRegression()

# 첫 번째, 두 번째 컬럼을 각각 2차원으로 reshape
model.fit(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1)) 

# coef_는 2차원 형태이므로 ravel()로 1차원으로 바꿔줌
plt.plot(x[:,0], x[:,0]*model.coef_.ravel() + model.intercept_, color='r')  # (x, y(weight&bias))
plt.show()
```


    
![png](/Machine-Learning/images/0401/output_6_0.png)
    


### 종속변수에 따라
- 연속적인 값 => Linear Regression
- binary classification => Logistic Regression

# Q. 분류 문제를 Linear Regression으로 해결할 수 있을까?


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Training Data Set
# 공부시간에 따른 합격 여부
x_data = np.array([1, 2, 5, 8, 10])  # 공부시간 (독립변수)
t_data = np.array([0, 0, 0, 1, 1])   # 합격여부 (종속변수, 0: 불합격, 1: 합격)

# model 생성
model = linear_model.LinearRegression()
# model 학습
model.fit(x_data.reshape(-1,1), t_data.reshape(-1,1))  # Multiple Linear Regression의 경우를 고려하여 2차원으로
# prediction
print(model.predict(np.array([[7]])))  # 공부시간이 7일 때 (2차원)

plt.scatter(x_data, t_data)
plt.plot(x_data, x_data*model.coef_.ravel() + model.intercept_)
plt.show()
```

    [[0.63265306]]
    


    
![png](/Machine-Learning/images/0401/output_9_1.png)
    


## [[0.63265306]] -> 0.5보다 크므로 합격일까?

### 데이터 바꿔서 해보기 


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Training Data Set
# 공부시간에 따른 합격 여부
x_data = np.array([1, 2, 5, 8, 10, 30])  # 공부시간 (독립변수)
t_data = np.array([0, 0, 0, 1, 1, 1])    # 합격여부 (종속변수, 0: 불합격, 1: 합격)

# model 생성
model = linear_model.LinearRegression()
# model 학습
model.fit(x_data.reshape(-1,1), t_data.reshape(-1,1))  # Multiple Linear Regression의 경우를 고려하여 2차원으로
# prediction
print(model.predict(np.array([[7]])))  # 공부시간이 7일 때 (2차원)

plt.scatter(x_data, t_data)
plt.plot(x_data, x_data*model.coef_.ravel() + model.intercept_)
plt.show()
```

    [[0.41831972]]
    


    
![png](/Machine-Learning/images/0401/output_12_1.png)
    


## [[0.63265306]] => [[0.41831972]] 위보다 작아짐
## => Linear Regression으로 판단/분류 문제를 해결하기에는 무리가 있다.

# Logistic Regression을 3가지 형태로 구현하기 🤔
- 간단한 Training Data Set 이용

## 1. python 구현


```python
import numpy as np

# 다변수 함수에 대한 수치미분을 수행하는 함수
def numerical_derivative(f,x):   
    
    delta_x = 1e-4
    derivative_x = np.zeros_like(x)  
    
    # iterator를 이용해서 입력변수 x에 대한 편미분 수행
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        
        idx = it.multi_index 
        tmp = x[idx]            
        x[idx] = tmp + delta_x   
        fx_plus_delta = f(x)
        
        x[idx] = tmp - delta_x    
        fx_minus_delta = f(x)
        
        derivative_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        
        x[idx] = tmp             
        it.iternext()
        
    return derivative_x

# Training Data Set
x_data = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(-1,1)  # Linear Regression을 위해 2차원으로
t_data = np.array([0,0,0,0,0,0,1,1,1,1]).reshape(-1,1)

# Weight, bias 정의
W = np.random.rand(1,1)
b = np.random.rand(1)

# logistic regression model, predict model, hypothesis
def predict(x):
    
    z = np.dot(x, W) + b          # linear regression model
    y = 1 / (1 + np.exp(-1 * z))  # logistic regression model
    
    result = 0
    
    # 계산되는 y값은 0과 1 사이의 확률값
    if y >= 0.5:
        result = 1
    else:
        result = 0
        
    return y, result

# Cross Entropy(log loss)
def loss_func(input_data):  # [W1, W2, W3, ... ,b]
    
    input_W = input_data[:-1].reshape(-1,1)
    input_b = input_data[-1]
    
    z = np.dot(x_data, input_W) + input_b
    y = 1 / (1 + np.exp(-1 * z))
    
    # y가 0이 되면 안되므로 코드에 영향을 미치지 않는 아주 작은 값을 더해준다.
    delta = 1e-7
    
    # cross entroy
    return -1 * np.sum(t_data * np.log(y + delta) + (1 - t_data) * np.log(1 - y + delta))
    
# learning rate 설정
learning_rate = 1e-4

# 반복학습 진행
for step in range(300000):
    
    input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)  # [W b]
    derivative_result = learning_rate * numerical_derivative(loss_func, input_param)

    W = W - derivative_result[:-1].reshape(1,1)
    b = b - derivative_result[-1]
    
    if step % 30000 == 0:
        input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)
        print('W : {}, b:{}, loss:{}'.format(W, b, loss_func(input_param)))
```

    W : [[0.42800666]], b:[0.39021404], loss:20.7741281875545
    W : [[0.26384694]], b:[-3.10594358], loss:2.9262521867333553
    W : [[0.37742307]], b:[-4.62696479], loss:2.1337004280208545
    W : [[0.45386132]], b:[-5.64769496], loss:1.7817370701211435
    W : [[0.51312912]], b:[-6.43715213], loss:1.5720154025549722
    W : [[0.56232598]], b:[-7.091062], loss:1.4283522945679454
    W : [[0.60482481]], b:[-7.65492192], loss:1.3216139245243288
    W : [[0.64250818]], b:[-8.1541223], loss:1.23798743038805
    W : [[0.67654109]], b:[-8.60436806], loss:1.1699759996678094
    W : [[0.70769829]], b:[-9.01609655], loss:1.113112620097403
    


```python
# prediction
study_hour = np.array([[13]])  # 12시간 - 불합격, 14시간 - 합격
y_prob, result = predict(study_hour)
print('합격 확률 : {}, 합격여부 : {}'.format(y_prob, result))
```

    합격 확률 : [[0.5444273]], 합격여부 : 1
    

## 2. sklearn 구현


```python
from sklearn import linear_model

# Training Data Set
x_data = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(-1,1)
t_data = np.array([0,0,0,0,0,0,1,1,1,1]).reshape(-1,1)

# model 생성
model = linear_model.LogisticRegression()

# model 학습
model.fit(x_data, t_data)

# prediction
study_hour = np.array([[13]])  # 12시간 - 불합격, 14시간 - 합격
result = model.predict(study_hour)  # 최종결과만 알려줌
result_prob = model.predict_proba(study_hour)  # 불합격, 합격 확률
print('합격 확률 : {}, 합격여부 : {}'.format(result_prob, result))

# 0.49990609 => 0.5에 못 미침
```

    합격 확률 : [[0.50009391 0.49990609]], 합격여부 : [0]
    

## 3. Tensorflow 구현


```python
import tensorflow as tf

# Training Data Set
x_data = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(-1,1)
t_data = np.array([0,0,0,0,0,0,1,1,1,1]).reshape(-1,1)

# placeholder
X = tf.placeholder(shape=[None,1], dtype=tf.float32)
T = tf.placeholder(shape=[None,1], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([1,1]))
b = tf.Variable(tf.random.normal([1]))

# Model(Hypothesis)
logit = tf.matmul(X, W) + b  # linear regression model
H = tf.sigmoid(logit)        # logistic regression model

# loss function
# sigmoid_cross_entropy_with_logits():
# linear regression model을 이용해서 cross entropy 값을 구하는 함수
# 두 개의 인자 필요 (logits, labels)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,
                                                              labels=T))
# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

# Session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 반복학습 진행
for step in range(30000):
    
    _, W_val, b_val, loss_val = sess.run([train, W, b, loss],
                                         feed_dict={X: x_data,
                                                    T: t_data})
    if step % 3000 == 0:
        print('W : {}, b:{}, loss:{}'.format(W_val, b_val, loss_val))
```


    W : [[0.12902921]], b:[-0.9290906], loss:0.5256268382072449
    W : [[0.1322414]], b:[-1.3360596], loss:0.45846638083457947
    W : [[0.15852948]], b:[-1.6907678], loss:0.4162381589412689
    W : [[0.18189465]], b:[-2.005362], loss:0.38302838802337646
    W : [[0.20288943]], b:[-2.287653], loss:0.35629352927207947
    W : [[0.2219421]], b:[-2.5435991], loss:0.33431944251060486
    W : [[0.23938502]], b:[-2.7777612], loss:0.31592875719070435
    W : [[0.255477]], b:[-2.9936805], loss:0.30029383301734924
    W : [[0.27042216]], b:[-3.194123], loss:0.28682073950767517
    W : [[0.28438252]], b:[-3.3812919], loss:0.27507340908050537
    


```python
# prediction
study_hour = np.array([[13]])  # 12시간 - 불합격, 14시간 - 합격
result = sess.run(H, feed_dict={X: study_hour})
print('합격 확률 : {}'.format(result))
```

    합격 확률 : [[0.57698435]]
    
