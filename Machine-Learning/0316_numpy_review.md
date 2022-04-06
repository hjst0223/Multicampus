# Numpy module 
- 다차원배열, 2차원 matrix 연산을 쉽고 빠르게 하기 위해 사용

### ndarray의 사칙연산(일반적인 연산) - ndarray의 shape이 같아야 함


```python
import numpy as np

arr1 = np.array([1,2,3])  # (3,)
arr2 = np.array([4,5,6])  # (3,)
print(arr1 + arr2)
```

    [5 7 9]
    

### 만약 shape이 다르면 오류 발생


```python
# 만약 shape이 다르다면 오류
arr1 = np.array([1,2,3,4])  # (4,)
arr2 = np.array([4,5,6])    # (3,)
print(arr1 + arr2)   
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Input In [2], in <cell line: 4>()
          2 arr1 = np.array([1,2,3,4])  # (4,)
          3 arr2 = np.array([4,5,6])    # (3,)
    ----> 4 print(arr1 + arr2)
    

    ValueError: operands could not be broadcast together with shapes (4,) (3,) 


### 하지만 shape이 달라도 shape을 맞출 수 있으면 연산 가능
### broadcasting이 발생해서 두 개의 ndarray의 shape을 똑같이 맞춤


```python
import numpy as np

arr1 = np.array([1,2,3,4])  # (4,)
print(arr1 + 1)             # broadcasting
```

    [2 3 4 5]
    

### 오류 발생하는 경우


```python
import numpy as np

arr1 = np.array([1,2,3,4,5,6])  # (6,)
arr2 = np.array([4,5,6])        # (3,)
arr1 + arr2                     # 오류
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Input In [4], in <cell line: 5>()
          3 arr1 = np.array([1,2,3,4,5,6])  # (6,)
          4 arr2 = np.array([4,5,6])        # (3,)
    ----> 5 arr1 + arr2
    

    ValueError: operands could not be broadcast together with shapes (6,) (3,) 


### 연산 가능한 경우


```python
import numpy as np

arr1 = np.array([[1,2,3], [4,5,6]])  # (2,3)
arr2 = np.array([4,5,6])             # (3,)
arr1 + arr2                          # 가능 - arr2의 차원을 올려서 수행
```




    array([[ 5,  7,  9],
           [ 8, 10, 12]])



## 행렬곱연산(matrix 곱연산)


```python
import numpy as np

arr1 = np.array([[1,2,3], [4,5,6]])    # (2,3)
arr2 = np.array([[4,5], [6,7], [8,9]]) # (3,2)
# 행렬곱연산의 결과는 (2,2)
print(np.matmul(arr1,arr2))
```

    [[ 40  46]
     [ 94 109]]
    

## 전치행렬(transpose)


```python
import numpy as np

arr = np.array([[1,2,3], [4,5,6]])  # (2,3)
print(arr)

print(arr.T)  # (3,2)
```

    [[1 2 3]
     [4 5 6]]
    [[1 4]
     [2 5]
     [3 6]]
    

## iterator(반복자)를 이용한 반복문 처리
- for문을 이용해서 반복처리를 하는데 
- ndarray는 while문과 iterator를 이용해서 반복처리하는 방식을 선호

### 1차원 ndarray에 반복처리

1. for문을 이용한 반복처리


```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])  # (5,)

for tmp in arr:
    print(tmp)
```

    1
    2
    3
    4
    5
    

2. iterator를 이용한 반복처리


```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])  # (5,)

# iterator 객체를 하나 얻어옴
# iterator 객체로부터 index 추출 가능
my_iter = np.nditer(arr, flags=['c_index'])

while not my_iter.finished:
    
    idx = my_iter.index
    print(arr[idx])  
    my_iter.iternext()
```

    1
    2
    3
    4
    5
    

### 2차원 ndarray에 반복처리

1. for문을 이용한 반복처리


```python
import numpy as np

arr = np.array([[1,2,3], [4,5,6]])  # (2,3)

for tmp1 in range(arr.shape[0]):
    for tmp2 in range(arr.shape[1]):
        print(arr[tmp1, tmp2])
```

    1
    2
    3
    4
    5
    6
    

2. iterator를 이용한 반복처리


```python
import numpy as np

arr = np.array([[1,2,3], [4,5,6]])  # (2,3)

my_iter = np.nditer(arr, flags=['multi_index'])

while not my_iter.finished:
    
    idx = my_iter.multi_index
    print(idx)
    print(arr[idx])
    my_iter.iternext()
```

    (0, 0)
    1
    (0, 1)
    2
    (0, 2)
    3
    (1, 0)
    4
    (1, 1)
    5
    (1, 2)
    6
    

# 다양한 집계함수와 axis 개념


```python
import numpy as np

arr = np.arange(1,7,1).reshape(2,3)
print(arr)
```

    [[1 2 3]
     [4 5 6]]
    

## ndarray 데이터에 대한 집계


```python
print(np.sum(arr))    # Numpy가 제공하는 함수 이용
print(arr.sum())      # ndarray가 갖고 있는 메소드 이용
print(arr.mean())     # 평균
print(arr.max())      # 최댓값
print(arr.min())      # 최솟값
print(arr.argmax())   # 가장 큰 값의 인덱스
print(arr.std())      # 표준편차
```

    21
    21
    3.5
    6
    1
    5
    1.707825127659933
    

## axis 개념
- numpy는 집계함수를 이용할 때 axis를 명시하지 않으면 전체를 대상으로 연산 수행


```python
arr = np.arange(1,7,1).reshape(2,3)
print(arr)   
```

    [[1 2 3]
     [4 5 6]]
    


```python
print(arr.sum(axis=0))  # 행 
print(arr.sum(axis=1))  # 열
```

    [5 7 9]
    [ 6 15]
    

## 연습문제

### 10보다 큰 수의 개수 (Boolean Indexing)


```python
import numpy as np

arr = np.arange(1,17,1).reshape(4,4)
print(arr)
```

    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]
     [13 14 15 16]]
    


```python
print((arr > 10).sum())
```

    6
    
