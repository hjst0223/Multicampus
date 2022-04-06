* Data 분석이 어려운 이유 - 데이터 & Computing Resource가 많이 필요함
* 우리가 얻는 데이터는 대부분 raw data로, 머신러닝의 학습에 사용하기 위해서는 가공이 필요함
* pandas module : pandas의 데이터타입을 구성하고 있는 numpy module
* numpy는 하나의 자료구조(list, dict)를 제공 => ndarray라는 자료구조를 제공

# Numpy 🙂
* 수치계산을 용이하기 하기 위한 python module로, 대용량의 다차원 배열과 matrix(행렬)연산에 용이함
* Numerical Python => Numpy
* 단 하나의 자료구조를 제공 => ndarray(n-dimensional array)
* ndarray는 python의 list와 상당히 유사, 차원의 개념 
* list는 여러 다른 타입의 데이터가 저장될 수 있으나 ndarray는 무조건 같은 데이터 타입만 저장됨
* 외장 module이기 때문에 설치해서 사용 => conda install numpy

## Python의 list


```python
a = [1, 2.3, 3, 4, True, 'Hello']
print(a)   
print(type(a)) 
```

    [1, 2.3, 3, 4, True, 'Hello']
    <class 'list'>
    

## Numpy의 ndarray


```python
import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr)        # 1차원
print(type(arr))   
print(arr.dtype) 
```

    [1 2 3 4]
    <class 'numpy.ndarray'>
    int32
    

## 중첩 리스트


```python
# 중첩리스트
my_list = [[1, 2, 3],[4, 5, 6]]
print(my_list)
print(my_list[1][1])  
```

    [[1, 2, 3], [4, 5, 6]]
    5
    

## 다차원 배열


```python
import numpy as np

# ndarray
arr = np.array([[1, 2, 3],[4, 5, 6]])
print(arr)  
print(arr[1,1]) 
print(arr[1])   
print(arr[1][1])
print(arr.dtype) # Numpy가 알아서 설정함
```

    [[1 2 3]
     [4 5 6]]
    5
    [4 5 6]
    5
    int32
    

#### 기본적으로 ndarray는 요소의 데이터 타입을 이용해서 dtype을 설정한다.
#### 데이터 타입을 지정해서 ndarray를 생성할 수 있다.


```python
arr = np.array([[1, 2, 3],[4, 5, 6]],
               dtype=np.float64)
print(arr)
print(arr.dtype)
```

    [[1. 2. 3.]
     [4. 5. 6.]]
    float64
    

## ndarray의 차원 관련 속성


```python
my_list = [1, 2, 3, 4]
arr = np.array(my_list)
print(arr)
print(arr.dtype)   # ndarray의 data type
print(arr.ndim)    # ndim은 차원수를 숫자로 알려줌
print(arr.shape)   # shape은 차원과 요소수를 tuple로 알려줌
                   # 1차원이므로 tuple안에 요소가 1개이며 그 값(4)이 요소의 개수를 나타냄
```

    [1 2 3 4]
    int32
    1
    (4,)
    


```python
my_list = [[1, 2, 3], [4, 5, 6]]
arr = np.array(my_list)
print(arr.ndim)
print(arr.shape) 
```

    2
    (2, 3)
    

## (2,2,3) 형태의 ndarray (면, 행, 열)


```python
my_list = [[[1,2,3], 
            [1,2,3]], 
           [[1,2,3], 
            [1,2,3]]]
arr = np.array(my_list)
print(arr.shape)
```

    (2, 2, 3)
    

## ndarray의 type 변환시키기 - astype()


```python
import numpy as np

arr = np.array([1.2, 2.3, 3, 4, 5.7])
print(arr.dtype)

# 실수=>정수 변환 시 소수점은 버림
new_arr = arr.astype(np.int32) 
print(new_arr)

new_arr = arr.astype(np.str_)
print(new_arr)
```

    float64
    [1 2 3 4 5]
    ['1.2' '2.3' '3.0' '4.0' '5.7']
    

# ndarray를 만드는 다양한 함수들

### np.array(), np.ones(), np.zeros(), np.full(). np.empty()


```python
import numpy as np
```


```python
my_list = [1, 2, 3]
arr = np.array(my_list)  # list를 이용하여 ndarray 생성
```


```python
arr = np.zeros((3,4))    # shape은 tuple로 표현
print(arr)               # 주어진 shape에 대한 ndarray를 생성하고 0으로 채움
```

    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    


```python
arr = np.ones((2,3))   
print(arr)               # 주어진 shape에 대한 ndarray를 생성하고 1로 채움
```

    [[1. 1. 1.]
     [1. 1. 1.]]
    


```python
arr = np.empty((2,3,3))
print(arr)               # 주어진 shape에 대한 ndarray를 생성하고 값을 채우지 않음
                         # 안에 쓰레기값이 들어갈 수 있음
                         # 상대적으로 ndarray를 빠르게 생성할 수 있음
```

    [[[4.67296746e-307 1.69121096e-306 9.34609111e-307]
      [1.33511018e-306 1.33511969e-306 6.23037996e-307]
      [1.69121639e-306 8.45593934e-307 9.34605716e-307]]
    
     [[8.01097889e-307 2.22522868e-306 1.33511562e-306]
      [1.24611402e-306 1.78020848e-306 1.78018811e-306]
      [7.56593696e-307 8.34423493e-308 2.29179042e-312]]]
    


```python
arr = np.full((2,3), 
              7, 
              dtype=np.float64)        
print(arr)
```

    [[7. 7. 7.]
     [7. 7. 7.]]
    

# shape을 직접 지정하지 않고 다른 ndarray의 shape을 이용해서 ndarray를 생성할 수 있다.

### np.ones_like(), np.zeros_like(), np.full_like(), np.empty_like()


```python
import numpy as np
```


```python
arr = np.array([[1,2,3], [4,5,6]], 
               dtype=np.int32)
print(arr)
```

    [[1 2 3]
     [4 5 6]]
    


```python
new_arr = np.zeros_like(arr)
print(new_arr)
```

    [[0 0 0]
     [0 0 0]]
    


```python
new_arr = np.ones_like(arr)
print(new_arr)
```

    [[1 1 1]
     [1 1 1]]
    

### np.arange()

## python의 range


```python
a = range(1,10)
print(a)    # 실제 값을 하나씩 가지고 있지는 않음
```

    range(1, 10)
    

## numpy의 arange


```python
import numpy as np

arr = np.arange(1,10)   # 1부터 10까지(10은 포함X) 1씩 증가하는 값으로 구성된 ndarray로, 실제로 값을 가지고 있음
print(arr)

arr = np.arange(1.3, 10.1, 2)
print(arr)
```

    [1 2 3 4 5 6 7 8 9]
    [1.3 3.3 5.3 7.3 9.3]
    

### linspace
* np.linspace(start, stop, num)
* start : 시작숫자와 , stop : 끝 숫자 ==> 둘 다 포함
* num : 그 안에 균일한 간격으로 몇개의 숫자가 들어갈지를 나타내는 숫자.
* 균일한 간격의 데이터를 생성해서  ndarray를 만들어 내는 함수
* 원소간의 간격 => (stop-start) / (num-1)


```python
import numpy as np

arr = np.linspace(0,10,6)
print(arr)

arr = np.linspace(0,120,31)
print(arr)
```

    [ 0.  2.  4.  6.  8. 10.]
    [  0.   4.   8.  12.  16.  20.  24.  28.  32.  36.  40.  44.  48.  52.
      56.  60.  64.  68.  72.  76.  80.  84.  88.  92.  96. 100. 104. 108.
     112. 116. 120.]
    

# 랜덤값을 이용해서 ndarray를 만들어 내는 방법

### np.random.normal(), np.random.rand(), np.random.randn()
### np.random.randint(), np.random.random()

#### 1. np.random.normal() 
* 난수로 채운 ndarray 생성 
* 정규분포 실수 표본을 추출
* 평균, 표준편차값 필요
* 난수를 추출하고 그 난수가 정규분포를 따르는지 확인


```python
import numpy as np
import matplotlib.pyplot as plt
mean = 50
std = 2
arr = np.random.normal(mean, std, (100000,)) # 1차원
print(arr)

plt.hist(arr,bins=100) # bins - 해당 데이터가 몇 개 들어가 있는지
plt.show()
```

    [46.1926441  50.56425143 51.05174488 ... 52.07787016 49.11293605
     49.51922114]
    


    
![png](/Machine-Learning/images/0315/output_43_1.png)
    


#### 2. np.random.rand()
* 난수로 채운 ndarray 생성
* [0,1) 0부터 1사이(0은 포함, 1은 불포함)의 실수형 난수를 균등분포에서 추출한 후 ndarray 생성


```python
import numpy as np
import matplotlib.pyplot as plt
arr = np.random.rand(100000) # rand(데이터 개수)
print(arr)

plt.hist(arr,bins=100)
plt.show()
```

    [0.05905535 0.63560078 0.30858127 ... 0.75908012 0.63365186 0.31609398]
    


    
![png](/Machine-Learning/images/0315/output_45_1.png)
    


#### 3. np.random.randn()
* 난수로 채운 ndarray 생성
* 표준정규분포에서 난수 추출(정규분포에서 평균이 0이고 표준편차가 1인 정규분포를 표준정규분포)


```python
# np.random.randn(d1, d2, d3) => 인자의 개수가 차원을 정함
arr = np.random.randn(100000) 
print(arr)

plt.hist(arr,bins=100)
plt.show()
```

    [ 0.9130715  -2.06103953  0.16213785 ... -0.27233668  0.19084834
      0.90858966]
    


    
![png](/Machine-Learning/images/0315/output_47_1.png)
    


#### 4. np.random.randint()
* 난수로 채운 ndarray 생성
* 균등분포에서 정수형 난수 추출
* low와 high값을 줘야 함


```python
arr = np.random.randint(1, 100, (100000,))
print(arr)

plt.hist(arr,bins=100)
plt.show()
```

    [ 1 62 34 ...  1 65 21]
    


    
![png](/Machine-Learning/images/0315/output_49_1.png)
    


#### 5. np.random.random()
* 난수로 채운 ndarray 생성
* 균등분포에서 실수형 난수 추출
* 범위는 [0,1) 사이에서 난수 추출


```python
# np.random.random(shape)
arr = np.random.random((100000,))
print(arr)

plt.hist(arr,bins=100)
plt.show()
```

    [0.19050647 0.13705129 0.00454385 ... 0.36157793 0.91951194 0.05678176]
    


    
![png](/Machine-Learning/images/0315/output_51_1.png)
    


# 기억해야 할 random 관련 함수들 😊

#### 1. seed()
    * 랜덤값도 프로그램 알고리즘에 의해서 추출되는 값
    * 랜덤값을 도출하는 초기값을 고정하면 항상 같은 랜덤값을 얻을 수 있음
    * 난수(랜덤값)의 재현성 확보 가능


```python
import numpy as np
np.random.seed(1) # [6 9 6 1 1] 고정
arr = np.random.randint(1,10, (5,))
print(arr)
```

    [6 9 6 1 1]
    


```python
import numpy as np
np.random.seed(None)
arr = np.random.randint(1,10, (5,))
print(arr)
```

    [1 1 9 7 6]
    

#### 2. shuffle() - 데이터의 순서 바꿀때 


```python
arr = np.arange(1,10)
print(arr)
```

    [1 2 3 4 5 6 7 8 9]
    


```python
np.random.shuffle(arr)   # 원본이 변경됨, 복사본을 만들지 않음
print(arr)
```

    [8 1 7 9 5 3 6 4 2]
    

#### 3. choice() - sampling에서 사용 
    * np.random.choice(a, size, replace, p)
    * a : 배열(ndarray)
    * size : 숫자 (몇 개를 추출할 것인지)
    * replace : True면 추출했던 것을 다시 추출할 수 있음
    * p : 확률(각 데이터가 선택될 수 있는 확률)


```python
# arr = np.random.choice(np.array([1,2,3,4,5]), 3)
arr = np.random.choice(np.array([1,2,3,4,5]), 3, replace=True) # 위와 같음 
print(arr)
```

    [3 3 1]
    


```python
arr = np.random.choice(np.array([1,2,3,4,5]), 3, replace=False)
print(arr)
```

    [3 1 2]
    


```python
arr = np.random.choice(np.array([1,2,3,4,5]), 3, p=[0.1, 0.2, 0.2, 0.5, 0])
print(arr)
```

    [2 3 3]
    

# ndarray의 shape과 관련된 함수들 🤨
### reshape(), ravel(), resize()

#### reshape() 
새로운 ndarray를 생성하는 것이 아니라 shape을 바꿔서 
원래 ndarray에 데이터를 다르게 보여주는 view 생성


```python
arr = np.array([[1,2,3,4,5,6], [7,8,9,10,11,12]])
print(arr)        # 2차원의 ndarray
print(arr.shape)
```

    [[ 1  2  3  4  5  6]
     [ 7  8  9 10 11 12]]
    (2, 6)
    


```python
new_arr = arr.reshape(3,4)
print(new_arr)
```

    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
    


```python
new_arr[0,0] = 100
print(new_arr)
print(arr)
```

    [[100   2   3   4]
     [  5   6   7   8]
     [  9  10  11  12]]
    [[100   2   3   4   5   6]
     [  7   8   9  10  11  12]]
    

모양을 바꾼 새로운 ndarray를 생성하고 싶을 경우 copy() 이용


```python
arr = np.array([[1,2,3,4,5,6], [7,8,9,10,11,12]])
new_arr = arr.reshape(3,4).copy()
new_arr[0,0] = 100
print(new_arr)
print(arr)
```

    [[100   2   3   4]
     [  5   6   7   8]
     [  9  10  11  12]]
    [[ 1  2  3  4  5  6]
     [ 7  8  9 10 11 12]]
    

#### reshape() - 내가 원하는 shape의 view를 만들 수 있음
#### ravel() - 무조건 1차원의 view를 만듦


```python
arr = np.array([[1,2,3,4,5,6], [7,8,9,10,11,12]])
print(arr)        # 2차원의 ndarray
print(arr.shape)

new_arr = arr.ravel()  # 1차원으로 view 생성
# new_arr = arr.reshape(12)
print(new_arr)
```

    [[ 1  2  3  4  5  6]
     [ 7  8  9 10 11 12]]
    (2, 6)
    [ 1  2  3  4  5  6  7  8  9 10 11 12]
    

#### resize()
    * 복사본을 만들지 않고 원본 변경
    * 요소의 개수가 달라도 수행 가능


```python
arr = np.array([[1,2,3,4,5,6], [7,8,9,10,11,12]])
print(arr)        # 2차원의 ndarray
print(arr.shape)

# new_arr = arr.reshape(3,5)  # reshape은 요소의 개수가 맞지 않으면 view가 생성되지 않음
new_arr = arr.resize(3,2)
print(new_arr)  
print(arr)            
```

    [[ 1  2  3  4  5  6]
     [ 7  8  9 10 11 12]]
    (2, 6)
    None
    [[1 2]
     [3 4]
     [5 6]]
    

# indexing & slicing 😳


```python
arr = np.arange(10,15,1)
print(arr)
print(arr[0])    # 기본적인 index
print(arr[1:3])  # 기본적인 slicing은 python의 list와 동일
print(arr[1:-1])
```

    [10 11 12 13 14]
    10
    [11 12]
    [11 12 13]
    


```python
arr = np.arange(0,12).reshape(3,4)
print(arr)
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    


```python
print(arr[1,2])
print(arr[1])   
print(arr[1,:]) 
```

    6
    [4 5 6 7]
    [4 5 6 7]
    


```python
print(arr[1:,2:])
```

    [[ 6  7]
     [10 11]]
    

## Boolean Indexing & Fancy Indexing

### Boolean Indexing
* boolean mask를 이용해서 indexing하는 방법 (True만 가져옴)
* boolean mask - True, False로만 구성된 ndarray


```python
import numpy as np

np.random.seed(1)
arr = np.random.randint(0, 10, (10,))
print(arr)   # [5 8 9 5 0 0 1 7 6 9] 고정
```

    [5 8 9 5 0 0 1 7 6 9]
    


```python
# ndarray의 사칙연산 => + , - , * , /
print(arr + 10) 
print(arr % 2)
print(arr % 2 == 0)  
```

    [15 18 19 15 10 10 11 17 16 19]
    [1 0 1 1 0 0 1 1 0 1]
    [False  True False False  True  True False False  True False]
    


```python
my_mask = (arr % 2 == 0)
print(arr[my_mask]) 
```

    [8 0 0 6]
    

python list 연산


```python
list1 = [1,2,3]
list2 = [4,5,6]
print(list1 + list2)  # list 연결
```

    [1, 2, 3, 4, 5, 6]
    

* ndarray는 같은 shape에 대해서만 연산 가능
* ndarray는 만약 shape을 맞출 수 있으면 broadcasting 진행 (차원을 맞춤)


```python
arr1 = np.array([1,2,3])  
arr2 = np.array([4,5,6])
print(arr1 + arr2)

arr1 = np.array([1,2,3])
print(arr1 + 4)   
```

    [5 7 9]
    [5 6 7]
    

### Fancy Indexing
* ndarray에 index list을 전달해서 indexing하는 방법


```python
arr = np.array([1,2,3,4,5,6])
print(arr)
```

    [1 2 3 4 5 6]
    


```python
print(arr[3])       # indexing
print(arr[3:])      # slicing
print(arr[[3, 5]])  # fancy indexing
```

    4
    [4 5 6]
    [4 6]
    


```python
arr = np.arange(0,12,1).reshape(3,4)
print(arr)
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    


```python
print(arr[0:2, 1:])  
```

    [[1 2 3]
     [5 6 7]]
    


```python
print(arr[0:2, [1,3]])  # 행에 대해서는 slicing, 열에 대해서는 fancy indexing
```

    [[1 3]
     [5 7]]
    


```python
print(arr[[0,2], [1,3]])  # [[1 3] [9 11]] 이 안 나옴
```

    [ 1 11]
    

#### np.ix_() 함수 이용


```python
print(arr[np.ix_([0,2], [1,3])])  
```

    [[ 1  3]
     [ 9 11]]
    
