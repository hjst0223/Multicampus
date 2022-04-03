# 공분산 구하기 👽


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

scores_df['en_deviation'] = scores_df['english'] - scores_df['english'].mean()
scores_df['ma_deviation'] = scores_df['mathematics'] - scores_df['mathematics'].mean()
scores_df['product_deviation'] = scores_df['en_deviation'] * scores_df['ma_deviation']
display(scores_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>english</th>
      <th>mathematics</th>
      <th>en_deviation</th>
      <th>ma_deviation</th>
      <th>product_deviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>42</td>
      <td>65</td>
      <td>-13.0</td>
      <td>-6.4</td>
      <td>83.2</td>
    </tr>
    <tr>
      <th>B</th>
      <td>69</td>
      <td>80</td>
      <td>14.0</td>
      <td>8.6</td>
      <td>120.4</td>
    </tr>
    <tr>
      <th>C</th>
      <td>56</td>
      <td>63</td>
      <td>1.0</td>
      <td>-8.4</td>
      <td>-8.4</td>
    </tr>
    <tr>
      <th>D</th>
      <td>41</td>
      <td>63</td>
      <td>-14.0</td>
      <td>-8.4</td>
      <td>117.6</td>
    </tr>
    <tr>
      <th>E</th>
      <td>57</td>
      <td>76</td>
      <td>2.0</td>
      <td>4.6</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>F</th>
      <td>48</td>
      <td>60</td>
      <td>-7.0</td>
      <td>-11.4</td>
      <td>79.8</td>
    </tr>
    <tr>
      <th>G</th>
      <td>65</td>
      <td>81</td>
      <td>10.0</td>
      <td>9.6</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>H</th>
      <td>49</td>
      <td>66</td>
      <td>-6.0</td>
      <td>-5.4</td>
      <td>32.4</td>
    </tr>
    <tr>
      <th>I</th>
      <td>65</td>
      <td>78</td>
      <td>10.0</td>
      <td>6.6</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>J</th>
      <td>58</td>
      <td>82</td>
      <td>3.0</td>
      <td>10.6</td>
      <td>31.8</td>
    </tr>
  </tbody>
</table>
</div>



```python
print('covariance(공분산) : ', scores_df['product_deviation'].mean())
```

    covariance(공분산) :  62.8
    

## Numpy의 함수로 공분산 구하기 - cov
- 결과는 covariance matrix(공분산 행렬)
- 편차의 곱의 평균을 구하는데 평균을 구할 때 n으로 나누는 경우는 ddof=0,  n-1로 나누는 경우는 ddof=1


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/student_scores_em.csv',
                 index_col='student number')

en_scores = df['english'][:10]
ma_scores = df['mathematics'][:10]

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



```python
# 0행 0열 : 영어와 영어의 공분산 => 영어의 분산
# 0행 1열 : 영어와 수학의 공분산
# 1행 0열 : 수학과 영어의 공분산
# 1행 1열 : 수학과 수학의 공분산 => 수학의 분산

cov_matrix = np.cov(en_scores, ma_scores, ddof=0)
print(cov_matrix)
```

    [[86.   62.8 ]
     [62.8  68.44]]
    

Pandas의 cov()는 무조건 n-1로 나누는 불편분산 형태로 사용된다.

## 실제 데이터를 이용하여 공분산 구하기

### KOSPI 지수와 삼성전자 주가에 대한 공분산
- 2018년도 기준으로 KOSPI 200안에서 삼성전자 차지하는 비율은 33%
- 삼성전자와 KOSPI는 양의 상관관계를 가지고 있을 것
- Yahoo finance(사이트)에서 주가 데이터를 받아서 사용


```python
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr  # module 설치
from datetime import datetime

start = datetime(2018,1,1)  # 특정 날짜에 대한 날짜 객체 생성
end = datetime(2018,12,31)   
```


```python
df_kospi = pdr.DataReader('^KS11', 'yahoo', start, end)
display(df_kospi)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-02</th>
      <td>2481.020020</td>
      <td>2465.939941</td>
      <td>2474.860107</td>
      <td>2479.649902</td>
      <td>262200</td>
      <td>2479.649902</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>2493.399902</td>
      <td>2481.909912</td>
      <td>2484.629883</td>
      <td>2486.350098</td>
      <td>331100</td>
      <td>2486.350098</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>2502.500000</td>
      <td>2466.449951</td>
      <td>2502.500000</td>
      <td>2466.459961</td>
      <td>333800</td>
      <td>2466.459961</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>2497.520020</td>
      <td>2475.510010</td>
      <td>2476.850098</td>
      <td>2497.520020</td>
      <td>308800</td>
      <td>2497.520020</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>2515.370117</td>
      <td>2494.179932</td>
      <td>2510.699951</td>
      <td>2513.280029</td>
      <td>311400</td>
      <td>2513.280029</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-12-21</th>
      <td>2061.510010</td>
      <td>2049.760010</td>
      <td>2052.699951</td>
      <td>2061.489990</td>
      <td>311400</td>
      <td>2061.489990</td>
    </tr>
    <tr>
      <th>2018-12-24</th>
      <td>2059.939941</td>
      <td>2046.180054</td>
      <td>2050.379883</td>
      <td>2055.010010</td>
      <td>285300</td>
      <td>2055.010010</td>
    </tr>
    <tr>
      <th>2018-12-26</th>
      <td>2037.829956</td>
      <td>2014.280029</td>
      <td>2028.810059</td>
      <td>2028.010010</td>
      <td>321500</td>
      <td>2028.010010</td>
    </tr>
    <tr>
      <th>2018-12-27</th>
      <td>2035.569946</td>
      <td>2021.390015</td>
      <td>2032.089966</td>
      <td>2028.439941</td>
      <td>398000</td>
      <td>2028.439941</td>
    </tr>
    <tr>
      <th>2018-12-28</th>
      <td>2046.969971</td>
      <td>2035.410034</td>
      <td>2036.699951</td>
      <td>2041.040039</td>
      <td>352700</td>
      <td>2041.040039</td>
    </tr>
  </tbody>
</table>
<p>244 rows × 6 columns</p>
</div>



```python
df_se = pdr.DataReader('005930.KS', 'yahoo', start, end)
display(df_se)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-02</th>
      <td>51400.0</td>
      <td>50780.0</td>
      <td>51380.0</td>
      <td>51020.0</td>
      <td>8474250.0</td>
      <td>45077.812500</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>52560.0</td>
      <td>51420.0</td>
      <td>52540.0</td>
      <td>51620.0</td>
      <td>10013500.0</td>
      <td>45607.925781</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>52180.0</td>
      <td>50640.0</td>
      <td>52120.0</td>
      <td>51080.0</td>
      <td>11695450.0</td>
      <td>45130.820312</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>52120.0</td>
      <td>51200.0</td>
      <td>51300.0</td>
      <td>52120.0</td>
      <td>9481150.0</td>
      <td>46049.691406</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>52520.0</td>
      <td>51500.0</td>
      <td>52400.0</td>
      <td>52020.0</td>
      <td>8383650.0</td>
      <td>45961.339844</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-12-21</th>
      <td>38650.0</td>
      <td>38100.0</td>
      <td>38200.0</td>
      <td>38650.0</td>
      <td>14947080.0</td>
      <td>34915.152344</td>
    </tr>
    <tr>
      <th>2018-12-24</th>
      <td>39050.0</td>
      <td>38300.0</td>
      <td>38500.0</td>
      <td>38800.0</td>
      <td>9729530.0</td>
      <td>35050.660156</td>
    </tr>
    <tr>
      <th>2018-12-26</th>
      <td>38750.0</td>
      <td>38300.0</td>
      <td>38400.0</td>
      <td>38350.0</td>
      <td>12707675.0</td>
      <td>34644.144531</td>
    </tr>
    <tr>
      <th>2018-12-27</th>
      <td>38800.0</td>
      <td>38100.0</td>
      <td>38700.0</td>
      <td>38250.0</td>
      <td>10510643.0</td>
      <td>34875.738281</td>
    </tr>
    <tr>
      <th>2018-12-28</th>
      <td>38900.0</td>
      <td>38200.0</td>
      <td>38250.0</td>
      <td>38700.0</td>
      <td>9900267.0</td>
      <td>35286.035156</td>
    </tr>
  </tbody>
</table>
<p>244 rows × 6 columns</p>
</div>



```python
cov_matrix = np.cov(df_kospi['Close'], df_se['Close'], ddof=0)
print(cov_matrix)  # 양의 상관관계
```

    [[   24078.14439225   488212.99831291]
     [  488212.99831291 11871059.41111261]]
    

### 남북경협주, 방산주를 이용해서 음의 상관관계에 대해 알아보기
- 남북경협주 => 부산산업 (011390.KS)
- 방산주 => LIG넥스원 (079550.KS)


```python
df_pusan = pdr.DataReader('011390.KS', 'yahoo', start, end)
display(df_pusan)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-02</th>
      <td>30000.0</td>
      <td>29450.0</td>
      <td>29500.0</td>
      <td>29900.0</td>
      <td>835.0</td>
      <td>29662.660156</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>31950.0</td>
      <td>29900.0</td>
      <td>29900.0</td>
      <td>31850.0</td>
      <td>4195.0</td>
      <td>31597.183594</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>34300.0</td>
      <td>31500.0</td>
      <td>31850.0</td>
      <td>32500.0</td>
      <td>7425.0</td>
      <td>32242.025391</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>32950.0</td>
      <td>30500.0</td>
      <td>32500.0</td>
      <td>32300.0</td>
      <td>4074.0</td>
      <td>32043.611328</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>33100.0</td>
      <td>32050.0</td>
      <td>32600.0</td>
      <td>32900.0</td>
      <td>3019.0</td>
      <td>32638.849609</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-12-21</th>
      <td>173000.0</td>
      <td>165500.0</td>
      <td>171000.0</td>
      <td>170000.0</td>
      <td>37740.0</td>
      <td>168650.578125</td>
    </tr>
    <tr>
      <th>2018-12-24</th>
      <td>175500.0</td>
      <td>166500.0</td>
      <td>170000.0</td>
      <td>172000.0</td>
      <td>36808.0</td>
      <td>170634.703125</td>
    </tr>
    <tr>
      <th>2018-12-26</th>
      <td>172500.0</td>
      <td>161000.0</td>
      <td>167500.0</td>
      <td>161000.0</td>
      <td>46252.0</td>
      <td>159722.031250</td>
    </tr>
    <tr>
      <th>2018-12-27</th>
      <td>167000.0</td>
      <td>162000.0</td>
      <td>163000.0</td>
      <td>162000.0</td>
      <td>24389.0</td>
      <td>160964.031250</td>
    </tr>
    <tr>
      <th>2018-12-28</th>
      <td>164500.0</td>
      <td>159500.0</td>
      <td>163000.0</td>
      <td>163000.0</td>
      <td>24845.0</td>
      <td>161957.640625</td>
    </tr>
  </tbody>
</table>
<p>244 rows × 6 columns</p>
</div>



```python
df_lig = pdr.DataReader('079550.KS', 'yahoo', start, end)
display(df_lig)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-02</th>
      <td>60500.0</td>
      <td>59500.0</td>
      <td>59800.0</td>
      <td>60300.0</td>
      <td>105294.0</td>
      <td>55506.976562</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>60300.0</td>
      <td>58800.0</td>
      <td>60300.0</td>
      <td>59200.0</td>
      <td>150080.0</td>
      <td>54494.414062</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>59500.0</td>
      <td>57700.0</td>
      <td>58600.0</td>
      <td>57800.0</td>
      <td>141177.0</td>
      <td>53205.695312</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>57500.0</td>
      <td>55400.0</td>
      <td>56100.0</td>
      <td>55800.0</td>
      <td>281124.0</td>
      <td>51364.667969</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>56000.0</td>
      <td>54000.0</td>
      <td>56000.0</td>
      <td>54500.0</td>
      <td>223472.0</td>
      <td>50168.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-12-21</th>
      <td>34350.0</td>
      <td>33200.0</td>
      <td>34000.0</td>
      <td>33550.0</td>
      <td>52369.0</td>
      <td>30883.236328</td>
    </tr>
    <tr>
      <th>2018-12-24</th>
      <td>33850.0</td>
      <td>33300.0</td>
      <td>33550.0</td>
      <td>33800.0</td>
      <td>26273.0</td>
      <td>31113.365234</td>
    </tr>
    <tr>
      <th>2018-12-26</th>
      <td>33750.0</td>
      <td>31800.0</td>
      <td>32500.0</td>
      <td>33350.0</td>
      <td>103487.0</td>
      <td>30699.134766</td>
    </tr>
    <tr>
      <th>2018-12-27</th>
      <td>34700.0</td>
      <td>32800.0</td>
      <td>33100.0</td>
      <td>33700.0</td>
      <td>69031.0</td>
      <td>31493.476562</td>
    </tr>
    <tr>
      <th>2018-12-28</th>
      <td>37750.0</td>
      <td>33650.0</td>
      <td>33650.0</td>
      <td>36950.0</td>
      <td>300059.0</td>
      <td>34530.683594</td>
    </tr>
  </tbody>
</table>
<p>244 rows × 6 columns</p>
</div>



```python
cov_matrix = np.cov(df_pusan['Close'], df_lig['Close'], ddof=0)
print(cov_matrix)  # 음의 상관관계
```

    [[ 4.62857448e+09 -3.84951773e+08]
     [-3.84951773e+08  6.33317924e+07]]
    

### KOSPI 지수와 삼성전자 주가에 대한 상관계수 구하기


```python
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr 
from datetime import datetime

start = datetime(2018,1,1)   # 특정 날짜에 대한 날짜 객체 생성
end = datetime(2018,12,31)   

df_kospi = pdr.DataReader('^KS11', 'yahoo', start, end)
df_se = pdr.DataReader('005930.KS', 'yahoo', start, end)

corr_coef = np.corrcoef(df_kospi['Close'], df_se['Close'])
print(corr_coef)
```

    [[1.         0.91317306]
     [0.91317306 1.        ]]
    
