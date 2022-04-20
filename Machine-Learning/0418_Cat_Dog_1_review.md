# 🐕 Dogs vs. Cats 🐈 data 활용하기
- [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv  # OpenCV
from sklearn import utils
from tqdm.notebook import tqdm  # progressbar 처리
import os  # file의 경로 설정 위해 필요
```


```python
# 폴더 경로 설정
train_dir = 'D:/jupyter_home/data/kaggle/cat_dog/train'

# label(target)을 알아내기 위한 함수
# cat => 0, dog => 1
# img는 파일 이름 (cat.0.jpg)
def labeling(img):
    class_name = img.split('.')[0]
    if class_name == 'cat':
        return 0
    if class_name == 'dog':
        return 1
    
# label data(t_data)와 pixel data(x_data)를 저장할 변수 만들기
x_data = []
t_data = []
```


```python
# 파일을 하나씩 반복하면서 처리
# 특정 폴더에 있는 모든 파일명을 알아야 함 => os.listdir()

for img in tqdm(os.listdir(train_dir),
                 total=len(os.listdir(train_dir)),
                 position=0,    # 시작 위치
                 leave=True):   # True: 지난 지점 칠하기
    
    label_data = labeling(img)  # 0 or 1
    
    img_path = os.path.join(train_dir, img)
    
    # img_path(이미지 full path)를 이용해서 opencv를 통해 픽셀 데이터 추출
    img_data = cv.resize(cv.imread(img_path, cv.IMREAD_GRAYSCALE), (80, 80))
    
    t_data.append(label_data)  # [0, 1, 1, 0, 0, ...]
    x_data.append(img_data.ravel())
```


![process_bar](/Machine-Learning/images/0418/bar.PNG)


```python
t_df = pd.DataFrame({
    'label': t_data
})
display(t_df.head())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
x_df = pd.DataFrame(x_data)
display(x_df.head())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>6390</th>
      <th>6391</th>
      <th>6392</th>
      <th>6393</th>
      <th>6394</th>
      <th>6395</th>
      <th>6396</th>
      <th>6397</th>
      <th>6398</th>
      <th>6399</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-87</td>
      <td>-82</td>
      <td>-76</td>
      <td>-72</td>
      <td>-68</td>
      <td>-65</td>
      <td>-60</td>
      <td>-58</td>
      <td>-55</td>
      <td>-53</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43</td>
      <td>43</td>
      <td>58</td>
      <td>45</td>
      <td>35</td>
      <td>28</td>
      <td>40</td>
      <td>52</td>
      <td>37</td>
      <td>41</td>
      <td>...</td>
      <td>38</td>
      <td>39</td>
      <td>44</td>
      <td>53</td>
      <td>59</td>
      <td>57</td>
      <td>70</td>
      <td>71</td>
      <td>47</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>38</td>
      <td>45</td>
      <td>46</td>
      <td>58</td>
      <td>57</td>
      <td>54</td>
      <td>59</td>
      <td>48</td>
      <td>51</td>
      <td>...</td>
      <td>88</td>
      <td>126</td>
      <td>-100</td>
      <td>-91</td>
      <td>-119</td>
      <td>-121</td>
      <td>-107</td>
      <td>-94</td>
      <td>-85</td>
      <td>-102</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-36</td>
      <td>-32</td>
      <td>-31</td>
      <td>-26</td>
      <td>-33</td>
      <td>-28</td>
      <td>-26</td>
      <td>-31</td>
      <td>-28</td>
      <td>-29</td>
      <td>...</td>
      <td>-20</td>
      <td>-24</td>
      <td>-28</td>
      <td>-52</td>
      <td>-52</td>
      <td>-53</td>
      <td>-40</td>
      <td>-35</td>
      <td>-44</td>
      <td>-43</td>
    </tr>
    <tr>
      <th>4</th>
      <td>114</td>
      <td>120</td>
      <td>115</td>
      <td>88</td>
      <td>96</td>
      <td>-117</td>
      <td>127</td>
      <td>-71</td>
      <td>-126</td>
      <td>-60</td>
      <td>...</td>
      <td>105</td>
      <td>101</td>
      <td>91</td>
      <td>95</td>
      <td>92</td>
      <td>97</td>
      <td>120</td>
      <td>89</td>
      <td>96</td>
      <td>84</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 6400 columns</p>
</div>



```python
# x_data와 t_data 결합
df = pd.merge(t_df, x_df, left_index=True, right_index=True)
display(df.head())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>6390</th>
      <th>6391</th>
      <th>6392</th>
      <th>6393</th>
      <th>6394</th>
      <th>6395</th>
      <th>6396</th>
      <th>6397</th>
      <th>6398</th>
      <th>6399</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-87</td>
      <td>-82</td>
      <td>-76</td>
      <td>-72</td>
      <td>-68</td>
      <td>-65</td>
      <td>-60</td>
      <td>-58</td>
      <td>-55</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>43</td>
      <td>43</td>
      <td>58</td>
      <td>45</td>
      <td>35</td>
      <td>28</td>
      <td>40</td>
      <td>52</td>
      <td>37</td>
      <td>...</td>
      <td>38</td>
      <td>39</td>
      <td>44</td>
      <td>53</td>
      <td>59</td>
      <td>57</td>
      <td>70</td>
      <td>71</td>
      <td>47</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>42</td>
      <td>38</td>
      <td>45</td>
      <td>46</td>
      <td>58</td>
      <td>57</td>
      <td>54</td>
      <td>59</td>
      <td>48</td>
      <td>...</td>
      <td>88</td>
      <td>126</td>
      <td>-100</td>
      <td>-91</td>
      <td>-119</td>
      <td>-121</td>
      <td>-107</td>
      <td>-94</td>
      <td>-85</td>
      <td>-102</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>-36</td>
      <td>-32</td>
      <td>-31</td>
      <td>-26</td>
      <td>-33</td>
      <td>-28</td>
      <td>-26</td>
      <td>-31</td>
      <td>-28</td>
      <td>...</td>
      <td>-20</td>
      <td>-24</td>
      <td>-28</td>
      <td>-52</td>
      <td>-52</td>
      <td>-53</td>
      <td>-40</td>
      <td>-35</td>
      <td>-44</td>
      <td>-43</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>114</td>
      <td>120</td>
      <td>115</td>
      <td>88</td>
      <td>96</td>
      <td>-117</td>
      <td>127</td>
      <td>-71</td>
      <td>-126</td>
      <td>...</td>
      <td>105</td>
      <td>101</td>
      <td>91</td>
      <td>95</td>
      <td>92</td>
      <td>97</td>
      <td>120</td>
      <td>89</td>
      <td>96</td>
      <td>84</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 6401 columns</p>
</div>



```python
# shuffle을 이용해서 dataframe의 row 섞기
shuffle_df = utils.shuffle(df)
display(shuffle_df.head())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>6390</th>
      <th>6391</th>
      <th>6392</th>
      <th>6393</th>
      <th>6394</th>
      <th>6395</th>
      <th>6396</th>
      <th>6397</th>
      <th>6398</th>
      <th>6399</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23858</th>
      <td>1</td>
      <td>-47</td>
      <td>-50</td>
      <td>-111</td>
      <td>-52</td>
      <td>-53</td>
      <td>100</td>
      <td>-13</td>
      <td>-97</td>
      <td>-96</td>
      <td>...</td>
      <td>48</td>
      <td>81</td>
      <td>99</td>
      <td>103</td>
      <td>84</td>
      <td>85</td>
      <td>-89</td>
      <td>-85</td>
      <td>-92</td>
      <td>-110</td>
    </tr>
    <tr>
      <th>13437</th>
      <td>1</td>
      <td>11</td>
      <td>6</td>
      <td>6</td>
      <td>30</td>
      <td>7</td>
      <td>11</td>
      <td>11</td>
      <td>22</td>
      <td>15</td>
      <td>...</td>
      <td>82</td>
      <td>84</td>
      <td>87</td>
      <td>90</td>
      <td>76</td>
      <td>123</td>
      <td>-117</td>
      <td>115</td>
      <td>-113</td>
      <td>-101</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0</td>
      <td>66</td>
      <td>73</td>
      <td>75</td>
      <td>71</td>
      <td>73</td>
      <td>78</td>
      <td>76</td>
      <td>74</td>
      <td>76</td>
      <td>...</td>
      <td>122</td>
      <td>114</td>
      <td>116</td>
      <td>119</td>
      <td>114</td>
      <td>104</td>
      <td>103</td>
      <td>103</td>
      <td>100</td>
      <td>91</td>
    </tr>
    <tr>
      <th>1410</th>
      <td>0</td>
      <td>-112</td>
      <td>-110</td>
      <td>-110</td>
      <td>-109</td>
      <td>-100</td>
      <td>-65</td>
      <td>-61</td>
      <td>-73</td>
      <td>-122</td>
      <td>...</td>
      <td>-65</td>
      <td>-63</td>
      <td>-63</td>
      <td>-62</td>
      <td>-71</td>
      <td>-69</td>
      <td>-69</td>
      <td>-68</td>
      <td>-67</td>
      <td>-69</td>
    </tr>
    <tr>
      <th>501</th>
      <td>0</td>
      <td>-127</td>
      <td>-116</td>
      <td>-116</td>
      <td>-105</td>
      <td>-111</td>
      <td>-104</td>
      <td>-104</td>
      <td>-115</td>
      <td>-114</td>
      <td>...</td>
      <td>34</td>
      <td>32</td>
      <td>28</td>
      <td>30</td>
      <td>27</td>
      <td>32</td>
      <td>33</td>
      <td>34</td>
      <td>32</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 6401 columns</p>
</div>



```python
# 최종적으로 만들어진 DataFrame을 파일로 저장
shuffle_df.to_csv('D:/jupyter_home/data/kaggle/cat_dog/train.csv', index=False)
```
