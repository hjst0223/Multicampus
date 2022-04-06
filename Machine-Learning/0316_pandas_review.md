### ndarray의 정렬, 연결, 삭제, CSV파일 로딩은 일반적으로 pandas로 처리한다.


```python
import numpy as np     # 관용적으로 numpy의 alias는 np를 사용
import pandas as pd    # 관용적으로 pandas의 alias는 pd를 사용
```

### ndarray 생성


```python
arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)
print(arr)
```

    [1. 2. 3. 4. 5.]
    

### Series 생성


```python
s = pd.Series([1, 2, 3, 4, 5], dtype=np.float64)
print(s)         # index와 value 출력
```

    0    1.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    dtype: float64
    


```python
print(s.values)  # ndarray
print(s.index)   # pandas의 객체
print(s.dtype)   # float64 (실수)
```

    [1. 2. 3. 4. 5.]
    RangeIndex(start=0, stop=5, step=1)
    float64
    


```python
import numpy as np     
import pandas as pd 

s = pd.Series([1, 2, 3, 4, 5],
              dtype=np.float64,
              index=['c', 'b', 'a', 'kk', '홍길동'])
print(s)
```

    c      1.0
    b      2.0
    a      3.0
    kk     4.0
    홍길동    5.0
    dtype: float64
    

### 만약 인덱스를 지정해서 Series를 만들면?
### => 숫자 index는 그대로 사용 가능 (내부에 숨어 있음)


```python
print(s[0])   # 숫자 index
print(s['c']) # 지정 index
```

    1.0
    1.0
    


```python
# Error
# print(s[100])   
# print(s['wow'])
# IndexError: index 100 is out of bounds for axis 0 with size 5
```

### 지정 index를 숫자로 사용하면?
### => 원래 index 사용 불가


```python
import numpy as np     
import pandas as pd 

s = pd.Series([1, 2, 3, 4, 5],
              dtype=np.float64,
              index=[0, 2, 100, 6, 9])
print(s)
```

    0      1.0
    2      2.0
    100    3.0
    6      4.0
    9      5.0
    dtype: float64
    


```python
print(s[0])
print(s[100])
```

    1.0
    3.0
    


```python
# KeyError
# print(s[1])
```

### 지정 index에 같은 index가 존재하면?
### => 모두 나옴


```python
import numpy as np     
import pandas as pd 

s = pd.Series([1, 2, 3, 4, 5],
              dtype=np.float64,
              index=['c', 'b', 'c', 'k', 'm'])
print(s)
```

    c    1.0
    b    2.0
    c    3.0
    k    4.0
    m    5.0
    dtype: float64
    


```python
print(s['c'])
```

    c    1.0
    c    3.0
    dtype: float64
    

## indexing & slicing
- indexing은 ndarray나 list와 비슷하나 slicing은 주의해야 함


```python
import numpy as np     
import pandas as pd 

s = pd.Series([1, 2, 3, 4, 5],
              dtype=np.float64,
              index=['c', 'b', 'a', 'k', 'm'])
print(s)
```

    c    1.0
    b    2.0
    a    3.0
    k    4.0
    m    5.0
    dtype: float64
    

### 숫자 index를 이용한 slicing(ndarray, list와 동일)


```python
print(s[0:3])
```

    c    1.0
    b    2.0
    a    3.0
    dtype: float64
    

### 지정 index를 사용한 slicing


```python
print(s['c':'a']) # 'a'까지 포함
```

    c    1.0
    b    2.0
    a    3.0
    dtype: float64
    

## boolean indexing, fancy indexing


```python
import numpy as np     
import pandas as pd 

s = pd.Series([1, 2, 3, 4, 5],
              dtype=np.float64,
              index=['c', 'b', 'a', 'k', 'm'])
print(s)
```

    c    1.0
    b    2.0
    a    3.0
    k    4.0
    m    5.0
    dtype: float64
    

### 짝수만 추출하는 boolean indexing


```python
print(s[s % 2 == 0])
```

    b    2.0
    k    4.0
    dtype: float64
    

### fancy indexing


```python
print(s[[0,2]])
```

    c    1.0
    a    3.0
    dtype: float64
    

## dictionary를 이용해서 Series 생성


```python
my_dict = {'서울': 1000, 
           '인천': 500,
           '제주': 200}

s = pd.Series(my_dict)
print(s)
```

    서울    1000
    인천     500
    제주     200
    dtype: int64
    

## 연습

### A공장의 2020년 1월 1일 부터 10일간 생산량의 Series로 저장
### 생산량: 평균 50, 표준편차 5인 정규분포에서 랜덤하게 생성(정수)
### index: 날짜


```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

start_day = datetime(2020,1,1)

# list comprehension => list 생성 시 제어문(for,if)을 이용.
s1 = pd.Series([int(x) for x in np.random.normal(50,5,(10,))],
               index=[start_day + timedelta(days=x) for x in range(10)])
print(s1)
```

    2020-01-01    52
    2020-01-02    51
    2020-01-03    49
    2020-01-04    53
    2020-01-05    52
    2020-01-06    53
    2020-01-07    48
    2020-01-08    47
    2020-01-09    48
    2020-01-10    55
    dtype: int64
    

### B공장의 2020년 1월 1일 부터 10일간 생산량의 Series로 저장
### 생산량: 평균 70, 표준편차 8인 정규분포에서 랜덤하게 생성(정수)
### index: 날짜


```python
s2 = pd.Series([int(x) for x in np.random.normal(70,8,(10,))],
               index=[start_day + timedelta(days=x) for x in range(10)])
print(s2)
```

    2020-01-01    61
    2020-01-02    77
    2020-01-03    61
    2020-01-04    85
    2020-01-05    57
    2020-01-06    71
    2020-01-07    68
    2020-01-08    65
    2020-01-09    58
    2020-01-10    57
    dtype: int64
    

### 날짜별 생산량의 합


```python
print(s1 + s2)
```

    2020-01-01    113
    2020-01-02    128
    2020-01-03    110
    2020-01-04    138
    2020-01-05    109
    2020-01-06    124
    2020-01-07    116
    2020-01-08    112
    2020-01-09    106
    2020-01-10    112
    dtype: int64
    

# 날짜 변경 후 생산량의 합 구하기

### A공장의 2020년 1월 1일 부터 10일간 생산량의 Series로 저장
### 생산량: 평균 50, 표준편차 5인 정규분포에서 랜덤하게 생성(정수)
### index: 날짜


```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

start_day_A_factory = datetime(2020,1,1)
start_day_B_factory = datetime(2020,1,5)

# list comprehension => list 생성 시 제어문(for,if)을 이용.
s1 = pd.Series([int(x) for x in np.random.normal(50,5,(10,))],
               index=[start_day_A_factory + timedelta(days=x) for x in range(10)])
print(s1)
```

    2020-01-01    41
    2020-01-02    55
    2020-01-03    43
    2020-01-04    53
    2020-01-05    41
    2020-01-06    43
    2020-01-07    54
    2020-01-08    50
    2020-01-09    53
    2020-01-10    52
    dtype: int64
    

### B공장의 2020년 1월 5일 부터 10일간 생산량의 Series로 저장
### 생산량: 평균 70, 표준편차 8인 정규분포에서 랜덤하게 생성(정수)
### index: 날짜


```python
s2 = pd.Series([int(x) for x in np.random.normal(70,8,(10,))],
               index=[start_day_B_factory + timedelta(days=x) for x in range(10)])
print(s2)
```

    2020-01-05    77
    2020-01-06    62
    2020-01-07    71
    2020-01-08    70
    2020-01-09    56
    2020-01-10    70
    2020-01-11    62
    2020-01-12    81
    2020-01-13    66
    2020-01-14    76
    dtype: int64
    

### 날짜별 생산량의 합 - Series는 인덱스가 같은 것끼리 연산


```python
print(s1 + s2)
```

    2020-01-01      NaN
    2020-01-02      NaN
    2020-01-03      NaN
    2020-01-04      NaN
    2020-01-05    118.0
    2020-01-06    105.0
    2020-01-07    125.0
    2020-01-08    120.0
    2020-01-09    109.0
    2020-01-10    122.0
    2020-01-11      NaN
    2020-01-12      NaN
    2020-01-13      NaN
    2020-01-14      NaN
    dtype: float64
    

# DataFrame

- Series의 집합
- dictionary를 이용해서 만듦


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['배수민', '장예은', '박시은', '심자윤'],
           '학년': [4, 3, 1, 2],
           '학점': [1.5, 2.4, 3.9, 3.2]}

df = pd.DataFrame(my_dict)
display(df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>학년</th>
      <th>학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>배수민</td>
      <td>4</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>장예은</td>
      <td>3</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>박시은</td>
      <td>1</td>
      <td>3.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>심자윤</td>
      <td>2</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>


## DataFrame의 기본 속성


```python
print(df.shape)  
print(df.values)  # 2차원 ndarray
print(df.size)    # DataFrame안의 요소 개수
print(df.ndim)   
print(df.index)   # 행 index 
print(df.columns) # 열 index
```

    (4, 3)
    [['배수민' 4 1.5]
     ['장예은' 3 2.4]
     ['박시은' 1 3.9]
     ['심자윤' 2 3.2]]
    12
    2
    RangeIndex(start=0, stop=4, step=1)
    Index(['이름', '학년', '학점'], dtype='object')
    


```python
df.index.name = '학번'
df.columns.name = '학생정보'
display(df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>학생정보</th>
      <th>이름</th>
      <th>학년</th>
      <th>학점</th>
    </tr>
    <tr>
      <th>학번</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>배수민</td>
      <td>4</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>장예은</td>
      <td>3</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>박시은</td>
      <td>1</td>
      <td>3.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>심자윤</td>
      <td>2</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>


### rename() - DataFrame의 column명과 index명 변경 


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['윤세은', '이채영', '심자윤', '장예은'],
           '학년': [4, 3, 1, 2],
           '학점': [1.5, 2.4, 3.9, 3.2]}

df = pd.DataFrame(my_dict)
display(df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>학년</th>
      <th>학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>윤세은</td>
      <td>4</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이채영</td>
      <td>3</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>심자윤</td>
      <td>1</td>
      <td>3.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>장예은</td>
      <td>2</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>


### inplace 옵션의 값이 True - 원본 수정, 복사본 만들지 않음
### inplace 옵션의 값이 False - 원본 그대로, 복사본 만듦


```python
new_df = df.rename(columns={'이름':'학생이름',
                            '학점':'평균평점'},
                   index={0:'one'},
                   inplace=False)
display(new_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학생이름</th>
      <th>학년</th>
      <th>평균평점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>윤세은</td>
      <td>4</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이채영</td>
      <td>3</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>심자윤</td>
      <td>1</td>
      <td>3.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>장예은</td>
      <td>2</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>


# set_index() - DataFrame의 특정 column을 index로 설정


```python
import numpy as np
import pandas as pd

my_dict = {'이름': ['심자윤', '윤세은', '배수민', '이채영'],
           '학년': [4, 3, 1, 2],
           '학점': [1.5, 2.4, 3.9, 3.2]}

df = pd.DataFrame(my_dict)
display(df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>학년</th>
      <th>학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>심자윤</td>
      <td>4</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>윤세은</td>
      <td>3</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>배수민</td>
      <td>1</td>
      <td>3.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이채영</td>
      <td>2</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>


### 이름 column을 index로 설정


```python
new_df = df.set_index('이름', 
                      inplace=False)
display(new_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학년</th>
      <th>학점</th>
    </tr>
    <tr>
      <th>이름</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>심자윤</th>
      <td>4</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>윤세은</th>
      <td>3</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>배수민</th>
      <td>1</td>
      <td>3.9</td>
    </tr>
    <tr>
      <th>이채영</th>
      <td>2</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>

