# 이상치 처리하기 🤔

## tukey fence 방식 이용하기


```python
import numpy as np
import matplotlib.pyplot as plt

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22.1])

figure = plt.figure()

ax1 = figure.add_subplot(1,2,1)
ax2 = figure.add_subplot(1,2,2)

ax1.set_title('Original Data Boxplot')
ax1.boxplot(data)

# numpy로 사분위 값을 구해서 Tukey Fence 방식으로 이상치 판단
first = np.percentile(data, 25)  # 1사분위 값
third = np.percentile(data, 75)  # 3사분위 값
IQR_value = third - first

upper_fence = third + (IQR_value * 1.5)
lower_fence = first - (IQR_value * 1.5)

# boolean indexing으로 이상치 추출
print(data[(data > upper_fence) | (data < lower_fence)])
# boolean indexing으로 이상치 제거 후 나머지 데이터 추출
result = data[(data <= upper_fence) & (data >= lower_fence)]
ax2.boxplot(result)
plt.show()
```

    [22.1]
    


    
![png](/Machine-Learning/images/0330/output_3_1.png)
    


## z-score(표준 점수) 이용하기


```python
import numpy as np
import matplotlib.pyplot as plt

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22.1])

mean = data.mean()  # 8.473333333333333
std = data.std()    # 5.331974826980671

zscore_data = (data - mean) / std

print(zscore_data)
```

    [-1.40160702 -1.21405925 -1.02651147 -0.8389637  -0.65141593 -0.46386816
     -0.27632038 -0.08877261  0.09877516  0.28632293  0.4738707   0.66141848
      0.84896625  1.03651402  2.55565098]
    


```python
# scipy는 sklearn과 유사한 통계전용 모듈
from scipy import stats

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22.1])

print(stats.zscore(data))  # 위 결과와 동일함
```

    [-1.40160702 -1.21405925 -1.02651147 -0.8389637  -0.65141593 -0.46386816
     -0.27632038 -0.08877261  0.09877516  0.28632293  0.4738707   0.66141848
      0.84896625  1.03651402  2.55565098]
    


```python
from scipy import stats

data = np.array([-10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22.1])

zscore_threshold = 2.0  # 일반적인 기준

outlier = data[np.abs(stats.zscore(data)) > zscore_threshold]

# 이상치 추출
print(outlier)

# boolean indexing으로 이상치 제거 후 나머지 데이터 추출 이상치 제거 후 나머지 데이터 추출
print(data[np.isin(data, outlier, invert=True)])
```

    [-10.   22.1]
    [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14.]
    

# 온도에 따른 오존량 예측하기 😃
- 결측치 처리
- 이상치 처리

## 1. Python 구현


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


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
training_data = df[['Ozone', 'Temp']]

# 결치값 처리
training_data.dropna(how='any', inplace=True)   # how='any' : 결치가 존재하는 행 삭제

# 이상치 처리
zscore_threshold = 2.0
outlier = training_data['Ozone'][(np.abs(stats.zscore(training_data['Ozone'])) > zscore_threshold)]
training_data = training_data.loc[np.isin(training_data['Ozone'],outlier, invert=True)]


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

# 반복 학습
for step in range(300000):
    
    input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)  # [W b]
    derivative_result = learning_rate * numerical_derivative(loss_func, input_param)

    W = W - derivative_result[0].reshape(1,1)
    b = b - derivative_result[1]
    
    if step % 30000 == 0:
        input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)
        print('W : {}, b:{}, loss:{}'.format(W, b, loss_func(input_param)))
```

    W : [[0.40912863]], b:[0.03163756], loss:541.2194916536396
    W : [[0.63400777]], b:[-10.55342513], loss:454.616788998746
    W : [[0.75753186]], b:[-20.24974225], loss:423.2520908599749
    W : [[0.87067233]], b:[-29.1309727], loss:396.93888466678305
    W : [[0.97430204]], b:[-37.26563386], loss:374.86359425219825
    W : [[1.06922049]], b:[-44.71648345], loss:356.34367526916986
    W : [[1.15615996]], b:[-51.54100366], loss:340.80651164903475
    W : [[1.23579117]], b:[-57.79184467], loss:327.771710925518
    W : [[1.30872846]], b:[-63.5172308], loss:316.8362510143927
    W : [[1.37553453]], b:[-68.76133255], loss:307.66201920177684
    

### 학습 종료 후 예측하기
- Temp : 62


```python
predict_data = predict(np.array([[62]]))
print('python으로 구한 온도가 62일때 오존량 : {}'.format(predict_data))
```

    python으로 구한 온도가 62일때 오존량 : [[15.51236159]]
    

### 그래프로 표현하기


```python
plt.scatter(x_data.ravel(), t_data.ravel())
plt.plot(x_data.ravel(), x_data.ravel()*W.ravel() + b, color='r')
plt.show()
```


    
![png](/Machine-Learning/images/0330/output_14_0.png)
    


## 2. sklearn 구현


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('./data/ozone.csv')
training_data = df[['Ozone', 'Temp']]
training_data.dropna(how='any', inplace=True)   # how='any' : 결치가 존재하는 행 삭제

# 이상치 처리
zscore_threshold = 2.0
outlier = training_data['Ozone'][(np.abs(stats.zscore(training_data['Ozone'])) > zscore_threshold)]
training_data = training_data.loc[np.isin(training_data['Ozone'],outlier, invert=True)]


# Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
t_data = training_data['Ozone'].values.reshape(-1,1)

# model 생성
model = linear_model.LinearRegression()

# model 학습
model.fit(x_data, t_data)

# 예측
result = model.predict(np.array([[62]]))
print('sklearn으로 구한 온도가 62도 일때의 오존량 : {}'.format(result))  
```

    sklearn으로 구한 온도가 62도 일때의 오존량 : [[4.51299041]]
    

### 그래프로 표현하기


```python
plt.scatter(x_data.ravel(), t_data.ravel())
plt.plot(x_data.ravel(), 
         x_data.ravel()*model.coef_.ravel() + model.intercept_, color='g')
plt.show()
```


    
![png](/Machine-Learning/images/0330/output_18_0.png)
    


# 정규화 작업하기 🐻
## Min-Max Scaling 


```python
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic[['age', 'fare']]

display(df.head())

def min_max(s):
    return (s - s.min()) / (s.max() - s.min())

result = df.apply(min_max, axis=0)

display(result.head())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.271174</td>
      <td>0.014151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.472229</td>
      <td>0.139136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.321438</td>
      <td>0.015469</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.434531</td>
      <td>0.103644</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.434531</td>
      <td>0.015713</td>
    </tr>
  </tbody>
</table>
</div>


# 온도에 따른 오존량 예측하기 😐
- 결측치 처리
- 이상치 처리
- 정규화 작업

## 1. Python 구현


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats  # 이상치 처리
from sklearn.preprocessing import MinMaxScaler  # 정규화 처리


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
training_data = df[['Ozone', 'Temp']]

# 결치값 처리
training_data.dropna(how='any', inplace=True)   # how='any' : 결치가 존재하는 행 삭제

# 이상치 처리
zscore_threshold = 2.0
outlier = training_data['Ozone'][(np.abs(stats.zscore(training_data['Ozone'])) > zscore_threshold)]
training_data = training_data.loc[np.isin(training_data['Ozone'],outlier, invert=True)]

# 정규화 처리
scaler_x = MinMaxScaler()   # 온도 처리를 위한 scaler(독립변수들을 위한 scaler)
scaler_t = MinMaxScaler()   # 오존량 처리를 위한 scaler(종속변수를 위한 scaler)

scaler_x.fit(training_data['Temp'].values.reshape(-1,1))
scaler_t.fit(training_data['Ozone'].values.reshape(-1,1))

scaled_Temp = scaler_x.transform(training_data['Temp'].values.reshape(-1,1))
scaled_Ozone = scaler_t.transform(training_data['Ozone'].values.reshape(-1,1))

training_data['Temp'] = scaled_Temp
training_data['Ozone'] = scaled_Ozone

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

# 반복 학습
for step in range(300000):
    
    input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)  # [W b]
    derivative_result = learning_rate * numerical_derivative(loss_func, input_param)

    W = W - derivative_result[0].reshape(1,1)
    b = b - derivative_result[1]
    
    if step % 30000 == 0:
        input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)
        print('W : {}, b:{}, loss:{}'.format(W, b, loss_func(input_param)))
```

    W : [[0.23234035]], b:[0.24655664], loss:0.0513074060067243
    W : [[0.38830938]], b:[0.18669836], loss:0.04156108433752865
    W : [[0.50092779]], b:[0.12680004], loss:0.03610648060840999
    W : [[0.58756085]], b:[0.08071387], loss:0.032878374306119985
    W : [[0.65420715]], b:[0.04526005], loss:0.03096793845091362
    W : [[0.70547773]], b:[0.01798564], loss:0.029837317332808325
    W : [[0.74491989]], b:[-0.0029964], loss:0.02916820085385308
    W : [[0.7752625]], b:[-0.01913775], loss:0.02877220890284499
    W : [[0.79860488]], b:[-0.0315552], loss:0.028537855647178586
    W : [[0.81656204]], b:[-0.04110786], loss:0.028399162301766815
    

### 학습 종료 후 예측하기


```python
predict_data = np.array([[62]])
scaled_predict_data = scaler_x.transform(predict_data)
python_result = predict(scaled_predict_data)

python_result = scaler_t.inverse_transform(python_result)

print('python으로 구한 온도가 62일때 오존량 : {}'.format(python_result))
```

    python으로 구한 온도가 62일때 오존량 : [[6.31269195]]
    


```python
python_x_data = x_data
python_t_data = t_data
```

## 2. sklearn 구현


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('./data/ozone.csv')
training_data = df[['Ozone', 'Temp']]
training_data.dropna(how='any', inplace=True)   # how='any' : 결치가 존재하는 행 삭제

# 이상치 처리
zscore_threshold = 2.0
outlier = training_data['Ozone'][(np.abs(stats.zscore(training_data['Ozone'])) > zscore_threshold)]
training_data = training_data.loc[np.isin(training_data['Ozone'],outlier, invert=True)]

# sklearn은 자체적으로 정규화 진행 - 따로 처리해줄 필요 없음

# Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
t_data = training_data['Ozone'].values.reshape(-1,1)

# model 생성
model = linear_model.LinearRegression()

# model 학습
model.fit(x_data, t_data)

# 예측
result = model.predict(np.array([[62]]))
print('sklearn으로 구한 온도가 62도 일때의 오존량 : {}'.format(result))  
```

    sklearn으로 구한 온도가 62도 일때의 오존량 : [[4.51299041]]
    


```python
sklearn_x_data = x_data
sklearn_t_data = t_data
```

## 그래프 비교하기


```python
figure = plt.figure()

python_ax = figure.add_subplot(1,2,1)

python_ax.scatter(python_x_data.ravel(), python_t_data.ravel())
python_ax.plot(python_x_data.ravel(), python_x_data.ravel()*W.ravel() + b, color='r')

sklearn_ax = figure.add_subplot(1,2,2)

sklearn_ax.scatter(sklearn_x_data.ravel(), sklearn_t_data.ravel())
sklearn_ax.plot(sklearn_x_data.ravel(), 
         sklearn_x_data.ravel()*model.coef_.ravel() + model.intercept_, color='g')

plt.show()
```


    
![png](/Machine-Learning/images/0330/output_31_0.png)
    


# quiz 점수에 따른 exam 점수 예측하기 😐

## 1. Python 구현


```python
import numpy as np
import pandas as pd


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


# Raw Data Loading
df = pd.read_csv('./data/student_exam_score.csv')

# 결측치, 이상치가 존재하지 않으므로 처리 X
# 각 변수의 scale이 비슷하기 때문에 정규화 작업 X

# Training Data Set
x_data = df.drop('exam', axis=1, inplace=False)
t_data = df.iloc[:,-1].values.reshape(-1,1)

# Weight, bias
W = np.random.rand(3,1)
b = np.random.rand(1)

# loss function
def loss_func(input_data):  # [w1  w2  w3  b]
    
    input_w = input_data[:-1].reshape(-1,1)
    input_b = input_data[-1:]
    
    y = np.dot(x_data,input_w) + input_b
    
    return np.mean(np.power(t_data-y,2))  # MSE(평균제곱오차)

# predict
def predict(x):
    y = np.dot(x,W) + b
    return y

# learning_rate
learning_rate = 1e-5

# 반복 학습
for step in range(300000):
    # 여기서 axis=0은 열을 뜻함
    input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)  # [w1 w2 w3 b]
    derivative_result = learning_rate * numerical_derivative(loss_func, input_param)

    W = W - derivative_result[:-1].reshape(-1,1)
    b = b - derivative_result[-1]
    
    if step % 30000 == 0:
        input_param = np.concatenate((W.ravel(), b.ravel()), axis=0)
        print('W : {}, b:{}, loss:{}'.format(W, b, loss_func(input_param)))
```

    W : [[0.34898076]
     [0.40307819]
     [0.24178501]], b:[0.38410774], loss:6934.315927622402
    W : [[0.35615991]
     [0.52930849]
     [1.12385273]], b:[0.33827718], loss:6.159138693493186
    W : [[0.35598764]
     [0.52954978]
     [1.12443175]], b:[0.28451313], loss:6.149501532388035
    W : [[0.35598698]
     [0.529699  ]
     [1.12492652]], b:[0.23136651], loss:6.140085326261593
    W : [[0.35598642]
     [0.52984645]
     [1.12541556]], b:[0.17883118], loss:6.13088448604833
    W : [[0.35598586]
     [0.52999221]
     [1.12589898]], b:[0.12690011], loss:6.121894085934448
    W : [[0.35598531]
     [0.53013629]
     [1.12637684]], b:[0.07556636], loss:6.1131093127687475
    W : [[0.35598477]
     [0.53027871]
     [1.1268492 ]], b:[0.02482305], loss:6.104525463485576
    W : [[0.35598424]
     [0.5304195 ]
     [1.12731612]], b:[-0.0253366], loss:6.0961379425870375
    W : [[0.35598371]
     [0.53055866]
     [1.12777768]], b:[-0.07491931], loss:6.087942259682693
    

 ### 학습 종료 후 예측하기


```python
result = predict(np.array([[89, 100, 95]]))
print(result)
```

    [[191.81041729]]
    

## 2. sklearn 구현


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# Raw Data Loading
df = pd.read_csv('./data/student_exam_score.csv')

# 결측치, 이상치가 존재하지 않으므로 처리 X
# sklearn은 자체적으로 정규화 진행 - 따로 처리해줄 필요 없음

# Training Data Set
x_data = df.drop('exam', axis=1, inplace=False)
t_data = df.iloc[:,-1].values.reshape(-1,1)

# model 생성
model = linear_model.LinearRegression()

# model 학습
model.fit(x_data, t_data)

# 예측
result = model.predict(np.array([[89, 100, 95]]))
print(result)
```

    [[192.50147537]]
    
