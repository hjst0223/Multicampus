```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Raw Data Loading
df = pd.read_csv('/content/drive/MyDrive/colab/mnist/train.csv')
display(df.head())
```

<pre>
   label  pixel0  pixel1  pixel2  pixel3  pixel4  pixel5  pixel6  pixel7  \
0      1       0       0       0       0       0       0       0       0   
1      0       0       0       0       0       0       0       0       0   
2      1       0       0       0       0       0       0       0       0   
3      4       0       0       0       0       0       0       0       0   
4      0       0       0       0       0       0       0       0       0   

   pixel8  ...  pixel774  pixel775  pixel776  pixel777  pixel778  pixel779  \
0       0  ...         0         0         0         0         0         0   
1       0  ...         0         0         0         0         0         0   
2       0  ...         0         0         0         0         0         0   
3       0  ...         0         0         0         0         0         0   
4       0  ...         0         0         0         0         0         0   

   pixel780  pixel781  pixel782  pixel783  
0         0         0         0         0  
1         0         0         0         0  
2         0         0         0         0  
3         0         0         0         0  
4         0         0         0         0  

[5 rows x 785 columns]
</pre>

```python
img_data = df.drop('label', axis=1, inplace=False).values

figure = plt.figure()
ax_arr = []

for n in range(10):
    ax_arr.append(figure.add_subplot(2,5,n+1))
    ax_arr[n].imshow(img_data[n].reshape(28,28),
                     cmap='Greys',
                     interpolation='nearest')
    
plt.tight_layout()
plt.show()
```
![png](/Machine-Learning/images/0412/output_4_0.png)
## 데이터 분할하기

- train 데이터와 validation 데이터



```python
train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(df.drop('label', axis=1, inplace=False),
                 df['label'],
                 test_size=0.3,
                 random_state=1,
                 stratify=df['label'])
```

## 정규화하기

- train_x_data, test_x_data에만



```python
scaler = MinMaxScaler()
scaler.fit(train_x_data)

norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)
```

### one-hot encoding 처리할 필요 X => sparse 이용

cross entropy

1. 이항 분류

2. 다항 분류 - categorical_crossentropy

	- sparse_ categorical_crossentropy(one-hot encoding 안 하는 경우)


## Tensorflow 2.x로 구현하기



```python
model = Sequential()

model.add(Flatten(input_shape=(784,)))

model.add(Dense(units=256,
                activation='relu'))
model.add(Dense(units=128,
                activation='relu'))
model.add(Dense(units=10,
                activation='softmax'))

model.compile(optimizer=SGD(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss',  # monitoring할 값
                   min_delta=0.001,     # threshold
                   patience=5,          # 참는 횟수
                   mode='auto',
                   restore_best_weights=True)

model.fit(norm_train_x_data,
          train_t_data,
          epochs=5000,
          batch_size=100,
          validation_split=0.3,
          verbose=0,
          callbacks=[es])
```

```python
<keras.callbacks.History at 0x7f634980e8d0>
```

```python
result = model.evaluate(norm_test_x_data, test_t_data)
print(result)
#           loss              accuracy 
```


```python
394/394 [==============================] - 1s 2ms/step - loss: 0.2101 - accuracy: 0.9417
[0.21013416349887848, 0.9416666626930237]
```
