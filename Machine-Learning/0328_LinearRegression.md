# DataFrame 만들기


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({'공부시간(x)': [1,2,3,4,5,7,8,10,12,13,14,15,18,20,25,28,30],
                   '시험점수(t)': [5,7,20,31,40,44,46,49,60,62,70,80,85,91,92,97,98]})

display(df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>공부시간(x)</th>
      <th>시험점수(t)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>40</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>44</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>46</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>49</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12</td>
      <td>60</td>
    </tr>
    <tr>
      <th>9</th>
      <td>13</td>
      <td>62</td>
    </tr>
    <tr>
      <th>10</th>
      <td>14</td>
      <td>70</td>
    </tr>
    <tr>
      <th>11</th>
      <td>15</td>
      <td>80</td>
    </tr>
    <tr>
      <th>12</th>
      <td>18</td>
      <td>85</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20</td>
      <td>91</td>
    </tr>
    <tr>
      <th>14</th>
      <td>25</td>
      <td>92</td>
    </tr>
    <tr>
      <th>15</th>
      <td>28</td>
      <td>97</td>
    </tr>
    <tr>
      <th>16</th>
      <td>30</td>
      <td>98</td>
    </tr>
  </tbody>
</table>
</div>


## scatter로 표현하기


```python
plt.scatter(df['공부시간(x)'], df['시험점수(t)'])
plt.show()
```


    
![png](/Machine-Learning/images/0328/output_3_0.png)
    


## y = wx + b 형태의 직선 그리기


```python
plt.scatter(df['공부시간(x)'], df['시험점수(t)'])

plt.plot(df['공부시간(x)'], 5*df['공부시간(x)']-7, color='g')

plt.show()
```


    
![png](/Machine-Learning/images/0328/output_5_0.png)
    


### y = wx + b 에서 b는 상수이기 때문에 실제 loss function 그래프 모양에 크게 영향을 주지 않는다.
###  => y = wx 형태로 loss 함수를 정의하고 그래프를 그려보자.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(1,101,1)
t = np.arange(1,101,1)

W = np.arange(-10,13)

loss = []

for tmp in W:
    loss.append(np.power(t - tmp * x, 2).mean())
    
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.plot(W, loss)

plt.show()
```


    
![png](/Machine-Learning/images/0328/output_7_0.png)
    

