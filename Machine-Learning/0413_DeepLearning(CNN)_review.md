```python
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
```

- 이미지 처리 module 상당히 많이 존재
- 가장 대표적인 module => opencv
- Pillow - 쉽게 사용할 수 있는 module 중 하나

# Pillow를 이용한 이미지 처리 🐣


```python
img = Image.open('./images/justice.jpg')

# 해당 이미지 파일에 대한 이미지 객체를 들고 옴
print(type(img))

plt.imshow(img)
plt.show()
```

    <class 'PIL.JpegImagePlugin.JpegImageFile'>
    


    
![png](/Machine-Learning/images/0413/output_3_1.png)
    



```python
pixel = np.array(img)  # pillow 이미지 객체를 이용해서 ndarray 생성
print(pixel)
```

    [[[ 14  30  90]
      [ 14  30  90]
      [ 15  31  91]
      ...
      [ 12  25  78]
      [ 11  24  77]
      [ 11  24  77]]
    
     [[ 15  31  91]
      [ 15  31  91]
      [ 15  31  91]
      ...
      [ 12  25  78]
      [ 12  25  78]
      [ 11  24  77]]
    
     [[ 15  31  91]
      [ 15  31  91]
      [ 15  31  91]
      ...
      [ 12  25  78]
      [ 12  25  78]
      [ 12  25  78]]
    
     ...
    
     [[ 34  75 163]
      [ 35  76 164]
      [ 35  76 166]
      ...
      [ 27  66 143]
      [ 27  66 143]
      [ 27  66 141]]
    
     [[ 34  75 163]
      [ 34  75 163]
      [ 35  76 166]
      ...
      [ 27  66 143]
      [ 26  65 142]
      [ 26  65 140]]
    
     [[ 34  75 163]
      [ 34  75 163]
      [ 35  76 166]
      ...
      [ 27  66 143]
      [ 26  65 142]
      [ 26  65 140]]]
    


```python
print(pixel.shape)  # (세로, 가로, 색상(channel))                 
```

    (426, 640, 3)
    

- .png 파일 => channel이 4 (R, G, B, alpha) 값으로 구성
    - alpha는 투명도


```python
print(pixel[100, 100])  # 100행 100열
# RED GREEN BLUE (0~255)
```

    [ 30  71 161]
    

- cropping (이미지 잘라내기)
- crop(좌상, 우하)


```python
crop_img = img.crop((30,100,150,300))
plt.imshow(crop_img)
plt.show()
```


    
![png](/Machine-Learning/images/0413/output_9_0.png)
    


## Image Resize


```python
print(img.size)  # 가로와 세로의 픽셀 수
```

    (640, 426)
    


```python
resize_img = img.resize((int(img.size[0] / 8), int(img.size[1] / 8)))  # tuple 형태 (가로, 세로)
plt.imshow(resize_img)
plt.show()
# 이미지가 깨짐
```


    
![png](/Machine-Learning/images/0413/output_12_0.png)
    


## Image Rotate


```python
rotate_img = img.rotate(180)
plt.imshow(rotate_img)
plt.show()
```


    
![png](/Machine-Learning/images/0413/output_14_0.png)
    



```python
# 이미지 저장
rotate_img.save('./images/rotate_justice.jpg')
```

## 컬러 이미지를 흑백 이미지로 바꾸기


```python
color_img = Image.open('./images/fruits.jpg')
plt.imshow(color_img)
plt.show()
```


    
![png](/Machine-Learning/images/0413/output_17_0.png)
    



```python
# pixel data 추출
color_pixel = np.array(color_img)
print(color_pixel.shape)
```

    (426, 640, 3)
    

### grey-scaled image로 변환
- color image의 pixel 값을 조절해서 gray-scaled로 변환
- color image의 각 pixel을 구성하고 있는 RGB 각 값들의 평균값으로 이미지의 각 pixel의 RGB 값을 다시 설정


```python
gray_pixel = color_pixel.copy()
print(gray_pixel.shape)

for y in range(gray_pixel.shape[0]):
    for x in range(gray_pixel.shape[1]):
        #    1차원                     scalar
        gray_pixel[y,x] = int(np.mean(gray_pixel[y,x]))
        
plt.imshow(gray_pixel)
plt.show()
```

    (426, 640, 3)
    


    
![png](/Machine-Learning/images/0413/output_20_1.png)
    



```python
# 흑백 이미지는 2차원으로도 표현이 가능
gray_2d_pixel = gray_pixel[:,:,0]

print(gray_2d_pixel.shape)
```

    (426, 640)
    


```python
plt.imshow(gray_2d_pixel)
plt.show()
```


    
![png](/Machine-Learning/images/0413/output_22_0.png)
    



```python
plt.imshow(gray_2d_pixel, cmap='Greys')
plt.show()
```


    
![png](/Machine-Learning/images/0413/output_23_0.png)
    



```python
plt.imshow(gray_2d_pixel, cmap='Greys_r')
plt.show()
```


    
![png](/Machine-Learning/images/0413/output_24_0.png)
    


## Tensorflow로 4차원 데이터 구성하기


```python
import tensorflow as tf
```

### 입력 이미지의 형태
- (이미지 개수, height, width, channel)
- (1, 3, 3, 1)


```python
image = np.array([[1], [1], [1]])
print(image.shape)
```

    (3, 1)
    


```python
image = np.array([[[1], [1], [1]],
                  [[1], [1], [1]],
                  [[1], [1], [1]]])
print(image.shape)
```

    (3, 3, 1)
    


```python
image = np.array([[[[1], [1], [1]],
                   [[1], [1], [1]],
                   [[1], [1], [1]]]])
print(image.shape)
```

    (1, 3, 3, 1)
    


```python
image = np.array([[[[9], [8], [7]],
                   [[6], [5], [4]],
                   [[3], [2], [1]]]], dtype=np.float64)
print(image.shape)
```

    (1, 3, 3, 1)
    

### filter의 형태
- filter의 channel 수는 이미지의 channel 수와 같음
- (filter height, filter width, filter channel, filter 개수)
- (2, 2, 1, 1)


```python
weight = np.array([2])
print(weight.shape)
```

    (1,)
    


```python
weight = np.array([[2]])
print(weight.shape)
```

    (1, 1)
    


```python
weight = np.array([[[2]],
                   [[2]]])
print(weight.shape)
```

    (2, 1, 1)
    


```python
weight = np.array([[[[2]],
                    [[2]]],
                   [[[2]],
                    [[2]]]])
print(weight.shape)
```

    (2, 2, 1, 1)
    

- stride : 1
- padding : 'VALID'(no padding) or 'SAME' (convolution 결과가 원본 이미지의 크기와 같음)


```python
con2d = tf.nn.conv2d(image,
                     weight,  # filter
                     strides=[1, 1, 1, 1],  # 차원 맞추기
                     padding='VALID')
```


```python
# 결과: (이미지 개수, 이미지 height, 이미지 width, channel)
sess = tf.Session()

result = sess.run(con2d)
print(result)
```


    [[[[56.]
       [48.]]
    
      [[32.]
       [24.]]]]
    
