## python의 함수도 변수에 저장되는 개념이다.
- 일급함수(first-classes function)를 지원하는 프로그래밍 언어 - 자바스크립트, 파이썬
- c언어는 일급함수를 지원하지 않음 
- 일급함수는 함수가 변수에 저장될 수 있으며 함수 이름을 변수처럼 사용할 수 있고 함수를 다른 함수의 인자로 사용할 수 있음


```python
def my_func():
    print('Hello')
    
print(my_func)  # 함수이름이 변수명
```

    <function my_func at 0x000001DB8F0FE5E0>
    


```python
def my_func2(x):
    print(x + 10)
    
tmp = 20    
my_func2(tmp)
```

    30
    


```python
def my_func3(x):
    x(20)         # 인자로 들어온 x를 이용해 함수 호출
    
def my_func4(x):
    print(x + 30)

my_func3(my_func4)    
```

    50
    

# 미분값을 구하는 함수 만들기😎
- f(x) = x^2
- f'(x) = 2x
- f'(5) = 10


```python
def my_func(x):
    return x ** 2

# 미분을 수행하는 함수
def numerical_derivative(f,x):
    
    delta_x = 1e-4 # 극한에 해당하는 값 - 너무 작은 값을 사용하면 실수 계산 오류 발생
                   # 1e-4 정도의 값을 이용하면 적당한 수치미분 값을 구할 수 있음
    
    return (f(x + delta_x) - f(x - delta_x)) / (2 * delta_x)

# f'(5)
result = numerical_derivative(my_func, 5)
print(result)
```

    9.999999999976694
    

## 다변수 함수의 수치미분(Nemerical Differentiation)
- f(x,y) = 2x + 3xy + y^3
- f(a,b,c) = 3a + 3bc + b^2 + c^3
- f'(1.0, 2.0) = ?


```python
import numpy as np

# 이렇게 구현하면 다변수함수의 수치미분하는 코드를 2변수, 3변수, 4변수,...에 대해서 각각 구현해야 함
def my_func(x,y):
    return 2 * x + 3 * x * y + y ** 3
print(my_func(1.0, 2.0))
```

    16.0
    


```python
import numpy as np

def my_func(input_value):  # input_value = [x y]
    x = input_value[0]
    y = input_value[1]
    return 2 * x + 3 * x * y + y ** 3

# 다변수 함수에 대한 수치미분을 수행하는 함수
def numerical_derivative(f,x):       # x : ndarray [1.0  2.0]
    
    delta_x = 1e-4
    derivative_x = np.zeros_like(x)  # derivative_x : [0.0  0.0]
    
    # iterator를 이용해서 입력변수 x에 대한 편미분 수행
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        
        idx = it.multi_index 
        tmp = x[idx]              # tmp : 1.0 -> 2.0
        
        x[idx] = tmp + delta_x    # x : ndarray [1.0001  2.0] -> ndarray [1.0  2.0001] 
        fx_plus_delta = f(x)
        
        x[idx] = tmp - delta_x    # x : ndarray [0.9999  2.0] -> ndarray [1.0  1.9999] 
        fx_minus_delta = f(x)
            
        derivative_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        
        x[idx] = tmp              #  x : ndarray [1.0  2.0]  
        it.iternext()
        
    return derivative_x

result = numerical_derivative(my_func, np.array([1.0, 2.0]))
print(result)
```

    [ 8.         15.00000001]
    

### 4변수 함수의 수치미분
- f(w, x, y, z) = wx + xyz + 3w + zy^2
- f'(1.0, 2.0, 3.0, 4.0) = ?


```python
def my_func(input_value):    # [[1.0  2.0
                             #    3.0  4.0]]
    w = input_value[0,0]
    x = input_value[0,1]
    y = input_value[1,0]
    z = input_value[1,1]
    
    return (w * x) + (x * y * z) + (3 * w) + z * (y**2)

result = numerical_derivative(my_func, np.array([[1.0, 2.0], [3.0, 4.0]]))
print(result)
```

    [[ 5. 13.]
     [32. 15.]]
    
