# Python으로 Simple Linear Regression 구현하기 😣


```python
import numpy as np

# Training Data Set 준비
# x_data => 독립변수(공부시간)
x_data = np.array([1, 2, 3, 4, 5], dtype=np.float64).reshape(5,1)
# t_data => 정답데이터(시험점수)
t_data = np.array([3, 5, 7, 9, 11], dtype=np.float64).reshape(5,1)

# Weight & bias 정의
W = np.random.rand(1,1)  # 1행 1열짜리(값은 1개) ndarray를 만들고
                         # 0과 1사이의 균등분포에서 실수 난수 추출
b = np.random.rand(1)     

# predict function (예측 함수, 모델, hypothesis)
def predict(x):
    y = np.dot(x,W) + b
    
    return y

# loss function
def loss_func(input_data):  # loss 함수는 w와 b의 함수
                            # input_data =>  [W b]
    
    input_W = input_data[0].reshape(1,1)
    input_b = input_data[1]
    
    # 예측값
    y = np.dot(x_data,input_W) + input_b
    
    # MSE(평균제곱오차)
    return np.mean(np.power(t_data-y,2))


# 다변수 함수에 대한 수치미분을 수행하는 함수
def numerical_derivative(f,x):       # x : ndarray [1.0  2.0]
    
    delta_x = 1e-4
    derivative_x = np.zeros_like(x)  # derivative_x : [0.0  0.0]
    
    # iterator를 이용해서 입력변수 x에 대한 편미분 수행
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        
        idx = it.multi_index 
        tmp = x[idx]            # tmp : 1.0
        
        x[idx] = tmp + delta_x  # x : ndarray [1.0001  2.0]
        fx_plus_delta = f(x)
        
        x[idx] = tmp - delta_x  # x : ndarray [0.9999  2.0]  
        fx_minus_delta = f(x)
        
        derivative_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        
        x[idx] = tmp            #  x : ndarray [1.0  2.0]  
        it.iternext()
        
    return derivative_x

# learning rate 값 설정 - 학습이 진행되는 과정을 보면서 적절하게 수정
# 초기에는 1e-4, 1e-3 정도로 설정해서 사용
learning_rate = 1e-4

# 학습과정 진행
for step in range(300000):
    
    input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)  # [W b]
    derivative_result = learning_rate * numerical_derivative(loss_func, input_param)

    W = W - derivative_result[0].reshape(1,1)
    b = b - derivative_result[1]
    
    if step % 30000 == 0:
        input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)
        print('W : {}, b:{}, loss:{}'.format(W, b, loss_func(input_param)))
```

    W : [[0.48004985]], b:[0.29264637], loss:32.363935650389664
    W : [[2.02671433]], b:[0.90355278], loss:0.0016931387097876946
    W : [[2.00968806]], b:[0.96502302], loss:0.0002226782166688278
    W : [[2.00351342]], b:[0.98731545], loss:2.928619367812306e-05
    W : [[2.00127415]], b:[0.9953999], loss:3.851661617301118e-06
    W : [[2.00046208]], b:[0.99833175], loss:5.065628322144572e-07
    W : [[2.00016757]], b:[0.999395], loss:6.662213051946095e-08
    W : [[2.00006077]], b:[0.9997806], loss:8.762009355207204e-09
    W : [[2.00002204]], b:[0.99992043], loss:1.1523619453365784e-09
    W : [[2.00000799]], b:[0.99997114], loss:1.5155633825702596e-10
    

## 학습 종료 후 예측하기


```python
predict_date = predict(np.array([[6]]))
print('6시간 공부했을 때 점수 : ', predict_date)
```

    6시간 공부했을 때 점수 :  [[13.00000693]]
    

# sklearn으로 Simple Linear Regression 구현하기 🤨


```python
import numpy as np
from sklearn import linear_model  # sklearn 설치 

# Training Data Set
x_data = np.array([1, 2, 3, 4, 5], dtype=np.float64).reshape(5,1)
t_data = np.array([3, 5, 7, 9, 11], dtype=np.float64).reshape(5,1)

# model 생성(Simple Linear Model)
model = linear_model.LinearRegression()

# model 학습
model.fit(x_data, t_data)

print('W: {}, b: {}'.format(model.coef_, model.intercept_))

# model을 이용한 예측
print(model.predict(np.array([[6]])))
```

    W: [[2.]], b: [1.]
    [[13.]]
    

# Ozone량 예측 모델 만들기 😏
- Ozone(오존량) : 종속변수
- Solar.R(태양광세기), Wind(바람), Temp(온도) : 독립변수
- Simple Linear Regression이므로 독립변수 1개만 사용 => 3개의 독립변수 중 Temp만 사용
- 온도에 따른 오존량 예측 모델 만들기

## 1. Python 구현


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 다변수 함수에 대한 수치미분을 수행하는 함수
def numerical_derivative(f,x):       # x : ndarray [1.0  2.0]
    
    delta_x = 1e-4
    derivative_x = np.zeros_like(x)  # derivative_x : [0.0  0.0]
    
    # iterator를 이용해서 입력변수 x에 대한 편미분 수행
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        
        idx = it.multi_index 
        tmp = x[idx]            # tmp : 1.0
        
        x[idx] = tmp + delta_x  # x : ndarray [1.0001  2.0]
        fx_plus_delta = f(x)
        
        x[idx] = tmp - delta_x  # x : ndarray [0.9999  2.0]  
        fx_minus_delta = f(x)
        
        derivative_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        
        x[idx] = tmp            #  x : ndarray [1.0  2.0]  
        it.iternext()
        
    return derivative_x


# Raw Data Set Loading
df = pd.read_csv('./data/ozone.csv')
# display(df.head())

training_data = df[['Ozone', 'Temp']]
# display(training_data)
print(training_data.shape)  # (153, 2)

# 결치값 처리
training_data.dropna(how='any', inplace=True)  # how='any' : 결치가 존재하는 행 삭제

# Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
t_data = training_data['Ozone'].values.reshape(-1,1)

# Weight, bias
W = np.random.rand(1,1)
b = np.random.rand(1)

# loss function
def loss_func(input_data):
    W = input_data[0].reshape(1,1)
    b = input_data[1]
    
    y = np.dot(x_data,W) + b
    return np.mean(np.power(t_data-y,2))

# predict
def predict(x):
    y = np.dot(x,W) + b
    return y

# learning_rate
learning_rate = 1e-4

# 학습과정 진행
for step in range(300000):
    
    input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)  # [W b]
    derivative_result = learning_rate * numerical_derivative(loss_func, input_param)

    W = W - derivative_result[0].reshape(1,1)
    b = b - derivative_result[1]
    
    if step % 30000 == 0:
        input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)
        print('W : {}, b:{}, loss:{}'.format(W, b, loss_func(input_param)))
```

    (153, 2)
    W : [[0.58255823]], b:[0.89502333], loss:873.7397048431517
    W : [[0.71294764]], b:[-11.42338102], loss:819.1178219871285
    W : [[0.8558565]], b:[-22.71546398], loss:776.5805741102502
    W : [[0.98686218]], b:[-33.06700586], loss:740.834252736922
    W : [[1.10695612]], b:[-42.55634629], loss:710.7947096187999
    W : [[1.21704719]], b:[-51.25529983], loss:685.5508770767608
    W : [[1.31796855]], b:[-59.22969944], loss:664.3371362784469
    W : [[1.41048396]], b:[-66.53989472], loss:646.5100968259189
    W : [[1.49529357]], b:[-73.24120864], loss:631.5290834033376
    W : [[1.5730392]], b:[-79.38435618], loss:618.9397376645915
    

### 학습 종료 후 예측하기


```python
predict_data = predict(np.array([[62]]))
print('python으로 구한 온도가 62일때 오존량 : {}'.format(predict_data))
```

    python으로 구한 온도가 62일때 오존량 : [[16.93138375]]
    

### 그래프로 표현하기


```python
plt.scatter(x_data.ravel(), t_data.ravel())
plt.plot(x_data.ravel(), x_data.ravel() * W.ravel() + b, color='r')
plt.show()
```


    
![png](/Machine-Learning/images/0329/output_13_0.png)
    


## 2. sklearn 구현


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('./data/ozone.csv')
training_data = df[['Ozone', 'Temp']]
training_data.dropna(how='any', inplace=True)   # how='any' : 결치가 존재하는 행 삭제

# Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
t_data = training_data['Ozone'].values.reshape(-1,1)

# model 생성
model = linear_model.LinearRegression()

# model 학습
model.fit(x_data, t_data)

# 예측
result = model.predict(np.array([[62]]))
print('sklearn으로 구한 온도가 62도일 때의 오존량 : {}'.format(result))  
```

    sklearn으로 구한 온도가 62도일 때의 오존량 : [[3.58411393]]
    

### 그래프로 표현하기


```python
plt.scatter(x_data.ravel(), t_data.ravel())
plt.plot(x_data.ravel(), 
         x_data.ravel()*model.coef_.ravel() + model.intercept_, color='g')
plt.show()
```


    
![png](/Machine-Learning/images/0329/output_17_0.png)
    

