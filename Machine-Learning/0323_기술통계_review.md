# 학생 체력 측정 데이터 불러오기 🤾‍♀️


```python
import numpy as np
import pandas as pd

df = pd.read_csv('./data/student_sports_test.csv',
                 index_col='학생번호')

display(df)

print(df.shape)
# 10행 5열의 데이터 
# 5열(5변수,5차원) 데이터
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>학년</th>
      <th>악력</th>
      <th>윗몸일으키기</th>
      <th>점수</th>
      <th>순위</th>
    </tr>
    <tr>
      <th>학생번호</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>40.2</td>
      <td>34</td>
      <td>15</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>34.2</td>
      <td>14</td>
      <td>7</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>28.8</td>
      <td>27</td>
      <td>11</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>39.0</td>
      <td>27</td>
      <td>14</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>50.9</td>
      <td>32</td>
      <td>17</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>36.5</td>
      <td>20</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>36.6</td>
      <td>31</td>
      <td>13</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>49.2</td>
      <td>37</td>
      <td>18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>26.0</td>
      <td>28</td>
      <td>10</td>
      <td>8</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>47.4</td>
      <td>32</td>
      <td>16</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


    (10, 5)
    

# 학생 점수 데이터 - 기술통계 📖


```python
import numpy as np
import pandas as pd

df = pd.read_csv('./data/student_scores_em.csv',
                 index_col='student number')
display(df)
print(df.shape)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>english</th>
      <th>mathematics</th>
    </tr>
    <tr>
      <th>student number</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>65</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>63</td>
    </tr>
    <tr>
      <th>5</th>
      <td>57</td>
      <td>76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>48</td>
      <td>60</td>
    </tr>
    <tr>
      <th>7</th>
      <td>65</td>
      <td>81</td>
    </tr>
    <tr>
      <th>8</th>
      <td>49</td>
      <td>66</td>
    </tr>
    <tr>
      <th>9</th>
      <td>65</td>
      <td>78</td>
    </tr>
    <tr>
      <th>10</th>
      <td>58</td>
      <td>82</td>
    </tr>
    <tr>
      <th>11</th>
      <td>70</td>
      <td>94</td>
    </tr>
    <tr>
      <th>12</th>
      <td>47</td>
      <td>75</td>
    </tr>
    <tr>
      <th>13</th>
      <td>51</td>
      <td>70</td>
    </tr>
    <tr>
      <th>14</th>
      <td>64</td>
      <td>77</td>
    </tr>
    <tr>
      <th>15</th>
      <td>62</td>
      <td>84</td>
    </tr>
    <tr>
      <th>16</th>
      <td>70</td>
      <td>84</td>
    </tr>
    <tr>
      <th>17</th>
      <td>71</td>
      <td>82</td>
    </tr>
    <tr>
      <th>18</th>
      <td>68</td>
      <td>85</td>
    </tr>
    <tr>
      <th>19</th>
      <td>73</td>
      <td>90</td>
    </tr>
    <tr>
      <th>20</th>
      <td>37</td>
      <td>70</td>
    </tr>
    <tr>
      <th>21</th>
      <td>65</td>
      <td>78</td>
    </tr>
    <tr>
      <th>22</th>
      <td>65</td>
      <td>93</td>
    </tr>
    <tr>
      <th>23</th>
      <td>61</td>
      <td>77</td>
    </tr>
    <tr>
      <th>24</th>
      <td>52</td>
      <td>88</td>
    </tr>
    <tr>
      <th>25</th>
      <td>57</td>
      <td>82</td>
    </tr>
    <tr>
      <th>26</th>
      <td>57</td>
      <td>76</td>
    </tr>
    <tr>
      <th>27</th>
      <td>75</td>
      <td>87</td>
    </tr>
    <tr>
      <th>28</th>
      <td>61</td>
      <td>84</td>
    </tr>
    <tr>
      <th>29</th>
      <td>47</td>
      <td>77</td>
    </tr>
    <tr>
      <th>30</th>
      <td>54</td>
      <td>82</td>
    </tr>
    <tr>
      <th>31</th>
      <td>66</td>
      <td>91</td>
    </tr>
    <tr>
      <th>32</th>
      <td>54</td>
      <td>75</td>
    </tr>
    <tr>
      <th>33</th>
      <td>54</td>
      <td>76</td>
    </tr>
    <tr>
      <th>34</th>
      <td>42</td>
      <td>78</td>
    </tr>
    <tr>
      <th>35</th>
      <td>37</td>
      <td>57</td>
    </tr>
    <tr>
      <th>36</th>
      <td>79</td>
      <td>89</td>
    </tr>
    <tr>
      <th>37</th>
      <td>56</td>
      <td>83</td>
    </tr>
    <tr>
      <th>38</th>
      <td>62</td>
      <td>69</td>
    </tr>
    <tr>
      <th>39</th>
      <td>62</td>
      <td>86</td>
    </tr>
    <tr>
      <th>40</th>
      <td>55</td>
      <td>81</td>
    </tr>
    <tr>
      <th>41</th>
      <td>63</td>
      <td>81</td>
    </tr>
    <tr>
      <th>42</th>
      <td>57</td>
      <td>80</td>
    </tr>
    <tr>
      <th>43</th>
      <td>57</td>
      <td>79</td>
    </tr>
    <tr>
      <th>44</th>
      <td>67</td>
      <td>87</td>
    </tr>
    <tr>
      <th>45</th>
      <td>55</td>
      <td>84</td>
    </tr>
    <tr>
      <th>46</th>
      <td>45</td>
      <td>71</td>
    </tr>
    <tr>
      <th>47</th>
      <td>66</td>
      <td>80</td>
    </tr>
    <tr>
      <th>48</th>
      <td>55</td>
      <td>77</td>
    </tr>
    <tr>
      <th>49</th>
      <td>64</td>
      <td>83</td>
    </tr>
    <tr>
      <th>50</th>
      <td>66</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>


    (50, 2)
    


```python
# 영어성적 상위 10개로 Series 생성
scores = df['english'][:10]  

# Series를 이용해서 DataFrame 생성
scores_df = pd.DataFrame(scores) 
display(scores_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>english</th>
    </tr>
    <tr>
      <th>student number</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
    </tr>
    <tr>
      <th>5</th>
      <td>57</td>
    </tr>
    <tr>
      <th>6</th>
      <td>48</td>
    </tr>
    <tr>
      <th>7</th>
      <td>65</td>
    </tr>
    <tr>
      <th>8</th>
      <td>49</td>
    </tr>
    <tr>
      <th>9</th>
      <td>65</td>
    </tr>
    <tr>
      <th>10</th>
      <td>58</td>
    </tr>
  </tbody>
</table>
</div>


## 영어 성적의 평균 구하기


```python
print(scores.mean())     # Series의 메소드 이용
print(np.mean(scores))   # Numpy의 mean() 함수 이용
print(scores_df.mean())  # DataFrame의 메소드 이용 - 결과는 Series
```

    55.0
    55.0
    english    55.0
    dtype: float64
    

## 영어 성적의 중위값 구하기


```python
print(np.median(scores))  
```

    56.5
    

## 영어 성적의 최빈값 구하기


```python
print(pd.Series([1, 1, 1, 2, 2, 3]).mode()) 
```

    0    1
    dtype: int64
    

## 영어 성적의 편차(deviation) 구하기


```python
deviation = scores - np.mean(scores)
print(deviation)
print(deviation.mean())  # 편차의 평균은 당연히 0
```

    student number
    1    -13.0
    2     14.0
    3      1.0
    4    -14.0
    5      2.0
    6     -7.0
    7     10.0
    8     -6.0
    9     10.0
    10     3.0
    Name: english, dtype: float64
    0.0
    

## 영어 성적의 분산 구하기


```python
print(np.mean(deviation ** 2))  # 편차의 제곱의 평균
print(np.var(scores))           # Numpy로 구하기
print(scores_df.var())          # DataFrame 모든 column에 대해서 variance를 구함
```

    86.0
    86.0
    english    95.555556
    dtype: float64
    

### Numpy로 구한 분산과 Pandas로 구한 분산이 다름 😮
### => 표본분산 vs. 불편분산
- 불편분산 : n-1로 나누어 평균을 구함
- Pandas : 불편분산 사용
- Numpy : 표본분산 사용


```python
print(scores_df.var(ddof=0))  # ddof=0 옵션을 주면 표본분산으로 계산됨
```

    english    86.0
    dtype: float64
    

## 영어성적의 표준편차(standart deviation) 구하기


```python
print(np.sqrt(np.var(scores)))
print(np.std(scores))
```

    9.273618495495704
    9.273618495495704
    

## 사분위값 구하는 Numpy 함수 - percentile()


```python
import numpy as np
import pandas as pd

df = pd.read_csv('./data/student_scores_em.csv',
                 index_col='student number')

scores = df['english'][:10]

scores_df = pd.DataFrame(scores)
display(scores_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>english</th>
    </tr>
    <tr>
      <th>student number</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
    </tr>
    <tr>
      <th>5</th>
      <td>57</td>
    </tr>
    <tr>
      <th>6</th>
      <td>48</td>
    </tr>
    <tr>
      <th>7</th>
      <td>65</td>
    </tr>
    <tr>
      <th>8</th>
      <td>49</td>
    </tr>
    <tr>
      <th>9</th>
      <td>65</td>
    </tr>
    <tr>
      <th>10</th>
      <td>58</td>
    </tr>
  </tbody>
</table>
</div>



```python
Q1 = np.percentile(scores, 25)   # 1사분위
Q2 = np.percentile(scores, 50)   # 2사분위 = 중위값 = median
Q3 = np.percentile(scores, 75)   # 3사분위

IQR = Q3 - Q1
print(IQR)
```

    15.0
    

## 1변수(1차원) 데이터로 도수분포표를 DataFrame으로 만들기


```python
import numpy as np
import pandas as pd

df = pd.read_csv('./data/student_scores_em.csv',
                 index_col='student number')

# 영어성적을 가지고 있는 ndarray 생성
scores = df['english'].values
print(scores)
```

    [42 69 56 41 57 48 65 49 65 58 70 47 51 64 62 70 71 68 73 37 65 65 61 52
     57 57 75 61 47 54 66 54 54 42 37 79 56 62 62 55 63 57 57 67 55 45 66 55
     64 66]
    


```python
# 특정 구간에 몇 개의 데이터가 포함되는지를 알려주는 Numpy 함수 - histogram()
freq, tmp = np.histogram(scores, bins=10, range=(0,100))
print(tmp)   # 경계값을 구할 수 있음
print(freq)  # 각 bin안의 도수(frequency) 
```

    [  0.  10.  20.  30.  40.  50.  60.  70.  80.  90. 100.]
    [ 0  0  0  2  8 16 18  6  0  0]
    


```python
# 행 index 만들기
freq_class = [str(i) + '~' + str(i+10) for i in range(0,100,10)]

freq_dist_df = pd.DataFrame({'Frequency':freq},
                            index=freq_class)
display(freq_dist_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0~10</th>
      <td>0</td>
    </tr>
    <tr>
      <th>10~20</th>
      <td>0</td>
    </tr>
    <tr>
      <th>20~30</th>
      <td>0</td>
    </tr>
    <tr>
      <th>30~40</th>
      <td>2</td>
    </tr>
    <tr>
      <th>40~50</th>
      <td>8</td>
    </tr>
    <tr>
      <th>50~60</th>
      <td>16</td>
    </tr>
    <tr>
      <th>60~70</th>
      <td>18</td>
    </tr>
    <tr>
      <th>70~80</th>
      <td>6</td>
    </tr>
    <tr>
      <th>80~90</th>
      <td>0</td>
    </tr>
    <tr>
      <th>90~100</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# class mark
class_mark = [ (i + (i+10)) / 2 for i in range(0,100,10)]
freq_dist_df['class_mark'] = class_mark

# 상대 도수(relative frequency)
rel_freq = freq / freq.sum()
freq_dist_df['rel_freq'] = rel_freq

display(freq_dist_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Frequency</th>
      <th>class_mark</th>
      <th>rel_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0~10</th>
      <td>0</td>
      <td>5.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>10~20</th>
      <td>0</td>
      <td>15.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>20~30</th>
      <td>0</td>
      <td>25.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>30~40</th>
      <td>2</td>
      <td>35.0</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>40~50</th>
      <td>8</td>
      <td>45.0</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>50~60</th>
      <td>16</td>
      <td>55.0</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>60~70</th>
      <td>18</td>
      <td>65.0</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>70~80</th>
      <td>6</td>
      <td>75.0</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>80~90</th>
      <td>0</td>
      <td>85.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>90~100</th>
      <td>0</td>
      <td>95.0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 누적 상대 도수(cumulative relative frequency)
freq_dist_df['cum_rel_freq'] = np.cumsum(rel_freq) # Numpy의 누적합을 구하는 함수 - cumsum()

display(freq_dist_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Frequency</th>
      <th>class_mark</th>
      <th>rel_freq</th>
      <th>cum_rel_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0~10</th>
      <td>0</td>
      <td>5.0</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>10~20</th>
      <td>0</td>
      <td>15.0</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>20~30</th>
      <td>0</td>
      <td>25.0</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>30~40</th>
      <td>2</td>
      <td>35.0</td>
      <td>0.04</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>40~50</th>
      <td>8</td>
      <td>45.0</td>
      <td>0.16</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>50~60</th>
      <td>16</td>
      <td>55.0</td>
      <td>0.32</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>60~70</th>
      <td>18</td>
      <td>65.0</td>
      <td>0.36</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>70~80</th>
      <td>6</td>
      <td>75.0</td>
      <td>0.12</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>80~90</th>
      <td>0</td>
      <td>85.0</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>90~100</th>
      <td>0</td>
      <td>95.0</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>


## 1변수(1차원) 데이터로 Histogram 그리기
- matplotlib의 hist()


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/student_scores_em.csv',
                 index_col='student number')

# 영어성적을 가지고 있는 ndarray 생성
scores = df['english'].values

# 그래프 그리기
figure = plt.figure(figsize=(10,6))

ax = figure.add_subplot(1,1,1) # 1행 1열 1번째 => 전체 캔버스에 꽉 차게

ax.hist(scores, bins=10, range=(0,100))

plt.show()
```


    
![png](/Machine-Learning/images/0323/output_29_0.png)
    


## 1변수(1차원) 데이터로 boxplot 그리기
- matplotlib 이용
- 데이터의 분포와 이상치 여부 알 수 있음


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/student_scores_em.csv',
                 index_col='student number')

# 영어성적을 가지고 있는 ndarray 생성 
scores = df['english'].values

fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(1,1,1)

ax.boxplot(scores)

plt.show()
```


    
![png](/Machine-Learning/images/0323/output_31_0.png)
    


## 2변수(2차원) 데이터로 수치지표와 그래프 표현하기


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/student_scores_em.csv',
                 index_col='student number')

display(df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>english</th>
      <th>mathematics</th>
    </tr>
    <tr>
      <th>student number</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>42</td>
      <td>65</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>63</td>
    </tr>
    <tr>
      <th>5</th>
      <td>57</td>
      <td>76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>48</td>
      <td>60</td>
    </tr>
    <tr>
      <th>7</th>
      <td>65</td>
      <td>81</td>
    </tr>
    <tr>
      <th>8</th>
      <td>49</td>
      <td>66</td>
    </tr>
    <tr>
      <th>9</th>
      <td>65</td>
      <td>78</td>
    </tr>
    <tr>
      <th>10</th>
      <td>58</td>
      <td>82</td>
    </tr>
    <tr>
      <th>11</th>
      <td>70</td>
      <td>94</td>
    </tr>
    <tr>
      <th>12</th>
      <td>47</td>
      <td>75</td>
    </tr>
    <tr>
      <th>13</th>
      <td>51</td>
      <td>70</td>
    </tr>
    <tr>
      <th>14</th>
      <td>64</td>
      <td>77</td>
    </tr>
    <tr>
      <th>15</th>
      <td>62</td>
      <td>84</td>
    </tr>
    <tr>
      <th>16</th>
      <td>70</td>
      <td>84</td>
    </tr>
    <tr>
      <th>17</th>
      <td>71</td>
      <td>82</td>
    </tr>
    <tr>
      <th>18</th>
      <td>68</td>
      <td>85</td>
    </tr>
    <tr>
      <th>19</th>
      <td>73</td>
      <td>90</td>
    </tr>
    <tr>
      <th>20</th>
      <td>37</td>
      <td>70</td>
    </tr>
    <tr>
      <th>21</th>
      <td>65</td>
      <td>78</td>
    </tr>
    <tr>
      <th>22</th>
      <td>65</td>
      <td>93</td>
    </tr>
    <tr>
      <th>23</th>
      <td>61</td>
      <td>77</td>
    </tr>
    <tr>
      <th>24</th>
      <td>52</td>
      <td>88</td>
    </tr>
    <tr>
      <th>25</th>
      <td>57</td>
      <td>82</td>
    </tr>
    <tr>
      <th>26</th>
      <td>57</td>
      <td>76</td>
    </tr>
    <tr>
      <th>27</th>
      <td>75</td>
      <td>87</td>
    </tr>
    <tr>
      <th>28</th>
      <td>61</td>
      <td>84</td>
    </tr>
    <tr>
      <th>29</th>
      <td>47</td>
      <td>77</td>
    </tr>
    <tr>
      <th>30</th>
      <td>54</td>
      <td>82</td>
    </tr>
    <tr>
      <th>31</th>
      <td>66</td>
      <td>91</td>
    </tr>
    <tr>
      <th>32</th>
      <td>54</td>
      <td>75</td>
    </tr>
    <tr>
      <th>33</th>
      <td>54</td>
      <td>76</td>
    </tr>
    <tr>
      <th>34</th>
      <td>42</td>
      <td>78</td>
    </tr>
    <tr>
      <th>35</th>
      <td>37</td>
      <td>57</td>
    </tr>
    <tr>
      <th>36</th>
      <td>79</td>
      <td>89</td>
    </tr>
    <tr>
      <th>37</th>
      <td>56</td>
      <td>83</td>
    </tr>
    <tr>
      <th>38</th>
      <td>62</td>
      <td>69</td>
    </tr>
    <tr>
      <th>39</th>
      <td>62</td>
      <td>86</td>
    </tr>
    <tr>
      <th>40</th>
      <td>55</td>
      <td>81</td>
    </tr>
    <tr>
      <th>41</th>
      <td>63</td>
      <td>81</td>
    </tr>
    <tr>
      <th>42</th>
      <td>57</td>
      <td>80</td>
    </tr>
    <tr>
      <th>43</th>
      <td>57</td>
      <td>79</td>
    </tr>
    <tr>
      <th>44</th>
      <td>67</td>
      <td>87</td>
    </tr>
    <tr>
      <th>45</th>
      <td>55</td>
      <td>84</td>
    </tr>
    <tr>
      <th>46</th>
      <td>45</td>
      <td>71</td>
    </tr>
    <tr>
      <th>47</th>
      <td>66</td>
      <td>80</td>
    </tr>
    <tr>
      <th>48</th>
      <td>55</td>
      <td>77</td>
    </tr>
    <tr>
      <th>49</th>
      <td>64</td>
      <td>83</td>
    </tr>
    <tr>
      <th>50</th>
      <td>66</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>



```python
en_scores = df['english'][:10]
ma_scores = df['mathematics'][:10]
print(en_scores)
print(ma_scores)
```

    student number
    1     42
    2     69
    3     56
    4     41
    5     57
    6     48
    7     65
    8     49
    9     65
    10    58
    Name: english, dtype: int64
    student number
    1     65
    2     80
    3     63
    4     63
    5     76
    6     60
    7     81
    8     66
    9     78
    10    82
    Name: mathematics, dtype: int64
    


```python
scores_df = pd.DataFrame({'english': en_scores.values,
                          'mathematics': ma_scores.values},
                         index=['A','B','C','D','E','F','G','H','I','J'])

display(scores_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>english</th>
      <th>mathematics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>42</td>
      <td>65</td>
    </tr>
    <tr>
      <th>B</th>
      <td>69</td>
      <td>80</td>
    </tr>
    <tr>
      <th>C</th>
      <td>56</td>
      <td>63</td>
    </tr>
    <tr>
      <th>D</th>
      <td>41</td>
      <td>63</td>
    </tr>
    <tr>
      <th>E</th>
      <td>57</td>
      <td>76</td>
    </tr>
    <tr>
      <th>F</th>
      <td>48</td>
      <td>60</td>
    </tr>
    <tr>
      <th>G</th>
      <td>65</td>
      <td>81</td>
    </tr>
    <tr>
      <th>H</th>
      <td>49</td>
      <td>66</td>
    </tr>
    <tr>
      <th>I</th>
      <td>65</td>
      <td>78</td>
    </tr>
    <tr>
      <th>J</th>
      <td>58</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>


### scatter 그리기


```python
figure = plt.figure(figsize=(4,4))
ax = figure.add_subplot(1,1,1)

ax.scatter(en_scores,ma_scores)
ax.set_xlabel('english')
ax.set_ylabel('mathematics')

plt.show()
```


    
![png](/Machine-Learning/images/0323/output_37_0.png)
    



```python
# 영어와 수학의 평균
print(en_scores.mean(), ma_scores.mean())  
```

    55.0 71.4
    
