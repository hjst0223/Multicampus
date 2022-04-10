# KNN 구현하기 😃
- BMI 예제(multinomial classification)
- Logistic Regression의 결과와 비교


```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Raw Data Loading
df = pd.read_csv('./data/bmi.csv', skiprows=3)

# Data Split
train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(df[['height', 'weight']],
                 df['label'],
                 test_size=0.3,
                 random_state=1,
                 stratify=df['label'])

# 결측치 X, 이상치 X
# 정규화 진행
scaler = MinMaxScaler()
scaler.fit(train_x_data)
norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)

# LogisticRegression 구현
model = LogisticRegression()
model.fit(norm_train_x_data, train_t_data)
acc = model.score(norm_test_x_data, test_t_data)  # scoring 기본값이 accuracy
print(f'LogisticRegression의 Accuracy : {acc}')

# KNN으로 구현
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(norm_train_x_data, train_t_data)
acc = knn_classifier.score(norm_test_x_data, test_t_data)
print(f'KNN의 Accuracy : {acc}')
```

    LogisticRegression의 Accuracy : 0.9851666666666666
    KNN의 Accuracy : 0.9985
    

# Ozone량 예측 Linear Regression 구현하기 😆
- Tensorflow 2.x
- 데이터 전처리


```python
%reset

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor  # 연속적인 숫자값 예측
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings(action='ignore')
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y
    


```python
# Raw Data Loading
df = pd.read_csv('./data/ozone.csv')
display(df.head())

x_data = df[['Solar.R', 'Wind', 'Temp']]  # DataFrame(2차원)
t_data = df['Ozone']                      # Series(1차원)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ozone</th>
      <th>Solar.R</th>
      <th>Wind</th>
      <th>Temp</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41.0</td>
      <td>190.0</td>
      <td>7.4</td>
      <td>67</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36.0</td>
      <td>118.0</td>
      <td>8.0</td>
      <td>72</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.0</td>
      <td>149.0</td>
      <td>12.6</td>
      <td>74</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18.0</td>
      <td>313.0</td>
      <td>11.5</td>
      <td>62</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.3</td>
      <td>56</td>
      <td>5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


## 1. 독립변수에 대한 Missing Value를 찾아 median으로 imputation
- np.nanmedian : nan을 제외하고 median 값 구함


```python
x_data.isnull().sum()
```




    Solar.R    7
    Wind       0
    Temp       0
    dtype: int64




```python
for col in x_data.columns:
    col_median = np.nanmedian(x_data[col])
    x_data[col].loc[x_data[col].isnull()] = col_median
    
x_data.isnull().sum()
```




    Solar.R    0
    Wind       0
    Temp       0
    dtype: int64



## 2. 독립변수에 대한 이상치를 검출한 후 이상치를 제외한 나머지 값들의 mean으로 이상치 대체


```python
zscore_threshold = 2.0

for col in x_data.columns:
    outlier = x_data[col][np.abs(stats.zscore(x_data[col])) > zscore_threshold]
    col_mean = np.mean(x_data.loc[~x_data[col].isin(outlier),col])
    x_data.loc[x_data[col].isin(outlier), col] = col_mean
```

## 3. 종속변수에 대한 이상치를 검출한 후 이상치를 제외한 나머지 값들의 mean으로 이상치 대체


```python
outlier = t_data[np.abs(stats.zscore(t_data)) > zscore_threshold]
col_mean = np.mean(~t_data.isin(outlier))
t_data[t_data.isin(outlier)] = col_mean
```

## 4. 정규화 진행


```python
scaler_x = MinMaxScaler()
scaler_t = MinMaxScaler()

scaler_x.fit(x_data.values)  
scaler_t.fit(t_data.values.reshape(-1,1))  # scaler는 2차원 ndarray로 사용해야 함

norm_x_data = scaler_x.transform(x_data.values)
norm_t_data = scaler_t.transform(t_data.values.reshape(-1,1)).ravel()
```

## 5. 종속변수의 Missing Value는 KNN을 이용해서 예측값 사용


```python
# 종속변수가 Missing Value가 아닌 독립변수들과 종속변수들을 추출(KNN을 학습하기 위해)
norm_train_x_data = norm_x_data[~np.isnan(norm_t_data)]
norm_train_t_data = norm_t_data[~np.isnan(norm_t_data)]

# 모델 생성
knn_regressor = KNeighborsRegressor(n_neighbors=2)
knn_regressor.fit(norm_train_x_data, norm_train_t_data)

# 종속변수가 Missing Value인 독립변수들을 입력으로 넣어 값 예측
knn_predict = knn_regressor.predict(norm_x_data[np.isnan(norm_t_data)])
norm_t_data[np.isnan(norm_t_data)] = knn_predict
```

- 최종적인 데이터 norm_x_data, norm_t_data

### sklearn 구현 & tensorflow 2.x 구현


```python
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

test_data = np.array([[330, 15, 80]])  # 태양광 330, 바람 15, 온도 80

# sklearn 구현
model = LinearRegression()
model.fit(norm_x_data, norm_t_data)
result = model.predict(scaler_x.transform(test_data))

print('sklearn 예측값 : {}'.format(scaler_t.inverse_transform(result.reshape(-1,1))))

# Tensorflow 2.x 구현(Linear Regression)
keras_model = Sequential()

keras_model.add(Flatten(input_shape=(3,)))   # Input Layer
keras_model.add(Dense(units=1,
                      activation='linear'))  # output Layer

keras_model.compile(optimizer=SGD(learning_rate=1e-2),
                    loss='mse')

keras_model.fit(norm_x_data,
                norm_t_data,
                epochs=5000,
                verbose=0)

result = keras_model.predict(scaler_x.transform(test_data))
print('tensorflow 예측값 : {}'.format(scaler_t.inverse_transform(result.reshape(-1,1))))
```

    sklearn 예측값 : [[36.93077619]]
    tensorflow 예측값 : [[36.97865]]
    

# Logistic Regression 😏
- binary classification을 skearn과 tensorflow 2.x로 구현
- titanic 데이터(Kaggle에서 다운로드)


```python
%reset

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 결측치 처리
# 실제 데이터이기 때문에 이상치가 검출되더라도 해당 값 사용
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y
    


```python
# Raw Data Loading
df = pd.read_csv('./data/kaggle/titanic/train.csv')

# 학습하기 좋은 데이터로 전처리

# 필요없는 column(feature) 삭제 (상관관계 분석)
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], 
             axis=1,
             inplace=False)

# 하나로 합칠 수 있는 컬럼은 합침
df['Family'] = df['SibSp'] + df['Parch']
df = df.drop(['SibSp', 'Parch'], axis=1, inplace=False)
display(df.head())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Embarked</th>
      <th>Family</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.isnull().sum()
```




    Survived      0
    Pclass        0
    Sex           0
    Age         177
    Embarked      2
    Family        0
    dtype: int64




```python
# 결측치 처리
# `Embarked` column은 결측치가 2개
# => 최빈값을 이용해서 missing value 채움 - 여기서는 Q
df['Embarked'] = df['Embarked'].fillna('Q')

# `Age` column의 결측치는 평균값으로 대체
df['Age'] = df['Age'].fillna(df['Age'].mean())
```


```python
# 문자로 되어 있는 값은 숫자로 변경
gender_string = { 'male': 0, 'female': 1 }
df['Sex'] = df['Sex'].map(gender_string)

embarked_string = { 'S': 0, 'C': 1, 'Q': 2 }
df['Embarked'] = df['Embarked'].map(embarked_string)

def age_category(age):
    if((age >= 0) & (age < 25)):
        return 0
    elif ((age >= 25) & (age < 50)):
        return 1
    else:
        return 2
    
df['Age'] = df['Age'].map(age_category)

df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Embarked</th>
      <th>Family</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Data Split
train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(df.drop('Survived', axis=1, inplace=False),
                 df['Survived'],
                 test_size=0.3,
                 random_state=1,
                 stratify=df['Survived'])

# Normalization
scaler = MinMaxScaler()
scaler.fit(train_x_data)

norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)
```


```python
# sklearn 구현
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(norm_train_x_data, train_t_data)
sklearn_result = model.score(norm_test_x_data, test_t_data)
print(f'sklearn 정확도 : {sklearn_result}')
```

    sklearn 정확도 : 0.7873134328358209
    


```python
# tensorflow 구현
keras_model = Sequential()
keras_model.add(Flatten(input_shape=(5,)))
keras_model.add(Dense(units=1,
                      activation='sigmoid'))
keras_model.compile(optimizer=SGD(learning_rate=1e-2),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
keras_model.fit(norm_train_x_data,
                train_t_data,
                epochs=1000,
                verbose=0)

keras_result = keras_model.evaluate(norm_test_x_data, test_t_data)
print(f'TF2.x 정확도 : {keras_result}')  
#                    loss                accuracy
```

    9/9 [==============================] - 0s 445us/step - loss: 0.4660 - accuracy: 0.7948
    TF2.x 정확도 : [0.46601223945617676, 0.7947761416435242]
    
