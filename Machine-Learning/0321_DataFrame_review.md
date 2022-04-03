# DataFrame 연결하기 - 행 방향 😐


```python
import numpy as np
import pandas as pd

df1 = pd.DataFrame({'a':['a0', 'a1', 'a2', 'a3'],
                    'b':[1, 2, 3, 4],
                    'c':['c0', 'c1', 'c2', 'c3']},
                   index=[0, 1, 2, 3])
display(df1)

df2 = pd.DataFrame({'b':[5, 6 , 7, 8],
                    'c':['c0', 'c1', 'c2', 'c3'],
                    'd':['d0', 'd1', 'd2', 'd3'],
                    'e':['e0', 'e1', 'e2', 'e3']},
                   index=[2, 3 , 4, 5])
display(df2)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a0</td>
      <td>1</td>
      <td>c0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1</td>
      <td>2</td>
      <td>c1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>3</td>
      <td>c2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>4</td>
      <td>c3</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>c0</td>
      <td>d0</td>
      <td>e0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>c1</td>
      <td>d1</td>
      <td>e1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>c2</td>
      <td>d2</td>
      <td>e2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>c3</td>
      <td>d3</td>
      <td>e3</td>
    </tr>
  </tbody>
</table>
</div>


## 1. 기존의 index 유지


```python
result_df = pd.concat([df1, df2], axis=0) 
display(result_df)   
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a0</td>
      <td>1</td>
      <td>c0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1</td>
      <td>2</td>
      <td>c1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>3</td>
      <td>c2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>4</td>
      <td>c3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>5</td>
      <td>c0</td>
      <td>d0</td>
      <td>e0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6</td>
      <td>c1</td>
      <td>d1</td>
      <td>e1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>7</td>
      <td>c2</td>
      <td>d2</td>
      <td>e2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>8</td>
      <td>c3</td>
      <td>d3</td>
      <td>e3</td>
    </tr>
  </tbody>
</table>
</div>


## 2.  기존의 index 없애고 숫자 index 다시 설정


```python
result_df = pd.concat([df1, df2], axis=0,
                      ignore_index=True) 
display(result_df)  
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a0</td>
      <td>1</td>
      <td>c0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1</td>
      <td>2</td>
      <td>c1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>3</td>
      <td>c2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>4</td>
      <td>c3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>5</td>
      <td>c0</td>
      <td>d0</td>
      <td>e0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>6</td>
      <td>c1</td>
      <td>d1</td>
      <td>e1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>7</td>
      <td>c2</td>
      <td>d2</td>
      <td>e2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>8</td>
      <td>c3</td>
      <td>d3</td>
      <td>e3</td>
    </tr>
  </tbody>
</table>
</div>


# DataFrame 연결하기 - 열 방향 🙃


```python
import numpy as np
import pandas as pd

df1 = pd.DataFrame({'a':['a0', 'a1', 'a2', 'a3'],
                    'b':[1, 2, 3, 4],
                    'c':['c0', 'c1', 'c2', 'c3']},
                   index=[0, 1, 2, 3])
display(df1)

df2 = pd.DataFrame({'b':[5, 6 , 7, 8],
                    'c':['c0', 'c1', 'c2', 'c3'],
                    'd':['d0', 'd1', 'd2', 'd3'],
                    'e':['e0', 'e1', 'e2', 'e3']},
                   index=[2, 3 , 4, 5])
display(df2)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a0</td>
      <td>1</td>
      <td>c0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1</td>
      <td>2</td>
      <td>c1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>3</td>
      <td>c2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>4</td>
      <td>c3</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>c0</td>
      <td>d0</td>
      <td>e0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>c1</td>
      <td>d1</td>
      <td>e1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>c2</td>
      <td>d2</td>
      <td>e2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>c3</td>
      <td>d3</td>
      <td>e3</td>
    </tr>
  </tbody>
</table>
</div>



```python
result_df = pd.concat([df1, df2],
                      axis=1)
display(result_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a0</td>
      <td>1.0</td>
      <td>c0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a1</td>
      <td>2.0</td>
      <td>c1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2</td>
      <td>3.0</td>
      <td>c2</td>
      <td>5.0</td>
      <td>c0</td>
      <td>d0</td>
      <td>e0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a3</td>
      <td>4.0</td>
      <td>c3</td>
      <td>6.0</td>
      <td>c1</td>
      <td>d1</td>
      <td>e1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>c2</td>
      <td>d2</td>
      <td>e2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>c3</td>
      <td>d3</td>
      <td>e3</td>
    </tr>
  </tbody>
</table>
</div>


# DataFrame의 결합(merge) 😏

## 1. inner join


```python
import numpy as np
import pandas as pd

data1 = {'학번':[1, 2, 3, 4],
         '이름':['재이', '세은', '아이사', '시은'],
         '학년':[3, 2, 1, 4]}

df1 = pd.DataFrame(data1)
display(df1)

data2 = {'학번':[1, 2, 3, 4],
         '학과':['철학과', '수학과', '컴퓨터', '국어국문'],
         '학점':[1.2, 3.3, 2.7, 4.0]}

df2 = pd.DataFrame(data2)
display(df2)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학번</th>
      <th>이름</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>재이</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>세은</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>아이사</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>시은</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학번</th>
      <th>학과</th>
      <th>학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>철학과</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>수학과</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>컴퓨터</td>
      <td>2.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>국어국문</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
result_df = pd.merge(df1, df2, on='학번', how='inner')  # inner join
display(result_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학번</th>
      <th>이름</th>
      <th>학년</th>
      <th>학과</th>
      <th>학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>재이</td>
      <td>3</td>
      <td>철학과</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>세은</td>
      <td>2</td>
      <td>수학과</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>아이사</td>
      <td>1</td>
      <td>컴퓨터</td>
      <td>2.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>시은</td>
      <td>4</td>
      <td>국어국문</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>


## 2. outer join


```python
import numpy as np
import pandas as pd

data1 = {'학번':[1, 2, 3, 4],
         '이름':['재이', '세은', '아이사', '시은'],
         '학년':[3, 2, 1, 4]}

df1 = pd.DataFrame(data1)
display(df1)

data2 = {'학번':[1, 2, 4, 5],
         '학과':['철학과', '수학과', '컴퓨터', '국어국문'],
         '학점':[1.2, 3.3, 2.7, 4.0]}

df2 = pd.DataFrame(data2)
display(df2)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학번</th>
      <th>이름</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>재이</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>세은</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>아이사</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>시은</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학번</th>
      <th>학과</th>
      <th>학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>철학과</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>수학과</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>컴퓨터</td>
      <td>2.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>국어국문</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>


### outer join


```python
result_df = pd.merge(df1, df2, on='학번', how='outer')  
display(result_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학번</th>
      <th>이름</th>
      <th>학년</th>
      <th>학과</th>
      <th>학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>재이</td>
      <td>3.0</td>
      <td>철학과</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>세은</td>
      <td>2.0</td>
      <td>수학과</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>아이사</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>시은</td>
      <td>4.0</td>
      <td>컴퓨터</td>
      <td>2.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>국어국문</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>


### left outer join


```python
result_df = pd.merge(df1, df2, on='학번', how='left')
display(result_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학번</th>
      <th>이름</th>
      <th>학년</th>
      <th>학과</th>
      <th>학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>재이</td>
      <td>3</td>
      <td>철학과</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>세은</td>
      <td>2</td>
      <td>수학과</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>아이사</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>시은</td>
      <td>4</td>
      <td>컴퓨터</td>
      <td>2.7</td>
    </tr>
  </tbody>
</table>
</div>


### right outer join


```python
result_df = pd.merge(df1, df2, on='학번', how='right')
display(result_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학번</th>
      <th>이름</th>
      <th>학년</th>
      <th>학과</th>
      <th>학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>재이</td>
      <td>3.0</td>
      <td>철학과</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>세은</td>
      <td>2.0</td>
      <td>수학과</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>시은</td>
      <td>4.0</td>
      <td>컴퓨터</td>
      <td>2.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>국어국문</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>


## 결합하려는 기준 column명이 서로 다른 경우


```python
import numpy as np
import pandas as pd

data1 = {'학번':[1, 2, 3, 4],
         '이름':['재이', '세은', '아이사', '시은'],
         '학년':[3, 2, 1, 4]}

df1 = pd.DataFrame(data1)
display(df1)

data2 = {'학생학번':[1, 2, 4, 5],
         '학과':['철학과', '수학과', '컴퓨터', '국어국문'],
         '학점':[1.2, 3.3, 2.7, 4.0]}

df2 = pd.DataFrame(data2)
display(df2)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학번</th>
      <th>이름</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>재이</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>세은</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>아이사</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>시은</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학생학번</th>
      <th>학과</th>
      <th>학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>철학과</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>수학과</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>컴퓨터</td>
      <td>2.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>국어국문</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
result_df = pd.merge(df1, df2, 
                     left_on='학번', 
                     right_on='학생학번', 
                     how='inner')  
display(result_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학번</th>
      <th>이름</th>
      <th>학년</th>
      <th>학생학번</th>
      <th>학과</th>
      <th>학점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>재이</td>
      <td>3</td>
      <td>1</td>
      <td>철학과</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>세은</td>
      <td>2</td>
      <td>2</td>
      <td>수학과</td>
      <td>3.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>시은</td>
      <td>4</td>
      <td>4</td>
      <td>컴퓨터</td>
      <td>2.7</td>
    </tr>
  </tbody>
</table>
</div>



## 결합하려는 기준 column과 다른 DataFrame의 행 index 결합하기


```python
import numpy as np
import pandas as pd

data1 = {'학번':[1, 2, 3, 4],
         '이름':['재이', '세은', '아이사', '시은'],
         '학년':[3, 2, 1, 4]}

df1 = pd.DataFrame(data1)
display(df1)

data2 = {'학과':['철학과', '수학과', '컴퓨터', '국어국문'],
         '학점':[1.2, 3.3, 2.7, 4.0]}

df2 = pd.DataFrame(data2,
                   index=[1, 2, 4, 5])
display(df2)
```

### df1의 학번과 df2의 행 index 결합


```python
result_df = pd.merge(df1, df2, 
                     left_on='학번', 
                     right_index=True, 
                     how='inner')  
display(result_df)
```

## 행 index로 결합하기


```python
import numpy as np
import pandas as pd

data1 = {'이름':['수민', '재이', '윤', '세은'],
         '학년':[3, 2, 1, 4]}

df1 = pd.DataFrame(data1,
                   index=[1, 2, 3, 4])
display(df1)

data2 = {'학과':['철학과', '수학과', '컴퓨터', '국어국문'],
         '학점':[1.2, 3.3, 2.7, 4.0]}

df2 = pd.DataFrame(data2,
                   index=[1, 2, 4, 5])
display(df2)
```

### df1의 행 index와 df2의 행 index 결합


```python
result_df = pd.merge(df1, df2, 
                     left_index=True, 
                     right_index=True, 
                     how='inner')  
display(result_df)
```

# 함수 mapping 🙄

## 1. Series 원소 각각에 함수 mapping - apply()
- titanic data set 이용

### 사용자 정의 함수 (일반적이지 않음)


```python
import numpy as np
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:,['age', 'fare']]
display(df.head())
```


```python
def add_10(n):
    return n + 10

def add_two_number(a,b):
    return a + b
```


```python
print(add_10(20))
```


```python
sr1 = df['age'].apply(add_10)
print(sr1.head())
```


```python
sr1 = df['age'].apply(add_two_number, b=30)
print(sr1.head())
```

### 람다식 사용 (일반적)


```python
import numpy as np
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:,['age', 'fare']]
display(df.head())
```


```python
sr1 = df['age'].apply(lambda x: x + 10)
print(sr1.head())
```

## 2. DataFrame 원소 각각에 함수 mapping - applymap()


```python
import numpy as np
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:,['age', 'fare']]
display(df.head())
```


```python
result_df = df.applymap(lambda x: x + 10)
display(result_df.head())
```


```python
result_df = df + 10
display(result_df.head())
```

## DataFrame의 행과 열에 함수 mapping
- 함수는 apply()
- axis 명시
- Min-Max Scaling 처리 : x(scaled) = (x - x(min)) / (x(max) - x(min))


```python
import numpy as np
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:,['age', 'fare']]
display(df.head())
```

DataFrame에 대한 정규화 진행


```python
def min_max(s):
    return (s - s.min()) / (s.max() - s.min())

# dataframe에 apply를 적용할 때 axis=0 하면 열 단위
result_df = df.apply(min_max, axis=0)
display(result_df.head())
```


```python
def add_two_number(a,b):
    return a + b

df['add'] = df.apply(lambda x: add_two_number(x['age'], x['fare']), 
                     axis=1)  # 행 단위
display(df.head())
```

# Grouping 🤓
복잡한 데이터를 어떤 기준에 따라서 여러 그룹으로 나누어 관찰하기 위한 방법

## 1. 1개의 column을 기준으로 grouping


```python
import numpy as np
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:,['age', 'sex', 'class', 'fare', 'survived']]
display(df.head())
```


```python
grouped = df.groupby(['class'])
print(grouped)
```

### group으로 묶인 모든 group 확인하기


```python
for key, group in grouped:
    print(key)
    display(group.head())
```

### 특정 group을 dataframe으로 가져오기


```python
result_df = grouped.get_group('Third')
display(result_df)
```

#### 일단 group으로 묶이게 되면 각 group에 대한 집계함수를 사용할 수 있다.


```python
my_mean = grouped.mean()
display(my_mean)
```

### group에 대한 filtering - fliter()


```python
import numpy as np
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:,['age', 'sex', 'class', 'fare', 'survived']]
display(df.head())
print(len(df))  # DataFrame에 대해 len()을 적용하면 행의 개수를 나타냄
```

각각의 그룹에 대한 len (First, Second, Third) => 300명 이상의 그룹만


```python
grouped = df.groupby(['class'])  
grouped_filter = grouped.filter(lambda x: len(x) > 300) 

display(grouped_filter.head())
```

## 2. 2개의 column을 기준으로 grouping


```python
import numpy as np
import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:,['age', 'sex', 'class', 'fare', 'survived']]
display(df.head())
```

### group으로 묶인 모든 group 확인하기


```python
grouped = df.groupby(['class', 'sex']) 

for key, group in grouped:
    print(key)
    display(group.head())
```

### 특정 group을 dataframe으로 가져오기


```python
my_group = grouped.get_group(('First', 'female'))
display(my_group)
```

### 각 group에 대한 집계함수 사용하기


```python
display(grouped.mean())
```
