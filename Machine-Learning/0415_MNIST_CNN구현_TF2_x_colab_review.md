# MNIST 예제로 CNN 구현하기 (TF 2.x) 👖



```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```


```python
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
## 데이터 전처리

- 결측치, 이상치, 정규화 => 결측치, 이상치 없음

- feature engineering

- train data와 test data 분리

  - => 학습을 위한 train data

  (train data와 validation data로 분리)

  - => 마지막 평가를 위한 test data (1번 사용)



```python
train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(df.drop('label', axis=1, inplace=False),
                 df['label'],
                 test_size=0.3,
                 random_state=1,
                 stratify=df['label'])

scaler = MinMaxScaler()
scaler.fit(train_x_data)
norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)
```

### t_data에 대한 one-hot encoding 처리는 하지 않아도 됨

=> keras 설정 이용


## Keras 구현



```python
# model 구현
model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1),
                 padding='valid',
                 strides=(1, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='valid',
                 strides=(1, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='valid',
                 strides=(1, 1)))

model.add(Flatten())
model.add(Dropout(rate=0.5))
model.add(Dense(units=256,
                activation='relu'))

model.add(Dense(units=10,
                activation='softmax'))

print(model.summary())
```

<pre>
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 3, 3, 64)          36928     
                                                                 
 flatten (Flatten)           (None, 576)               0         
                                                                 
 dropout (Dropout)           (None, 576)               0         
                                                                 
 dense (Dense)               (None, 256)               147712    
                                                                 
 dense_1 (Dense)             (None, 10)                2570      
                                                                 
=================================================================
Total params: 206,026
Trainable params: 206,026
Non-trainable params: 0
_________________________________________________________________
None
</pre>

```python
# model 실행 옵션
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```


```python
# model 학습
# norm_train_x_data - 이 학습 데이터의 일부를 validation data로 활용 
# => epoch마다 평가도 같이 진행
# 평가는 train에 대한 loss, accuracy, valid data에 대한 loss, accuracy
history = model.fit(norm_train_x_data.reshape(-1, 28, 28, 1),
                    train_t_data,
                    epochs=200,
                    batch_size=100,
                    verbose=1,
                    validation_split=0.3)
```

<pre>
Epoch 1/200
206/206 [==============================] - 12s 15ms/step - loss: 0.4934 - accuracy: 0.8428 - val_loss: 0.1364 - val_accuracy: 0.9562
Epoch 2/200
206/206 [==============================] - 3s 12ms/step - loss: 0.1340 - accuracy: 0.9567 - val_loss: 0.0759 - val_accuracy: 0.9769
Epoch 3/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0928 - accuracy: 0.9705 - val_loss: 0.0675 - val_accuracy: 0.9785
Epoch 4/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0735 - accuracy: 0.9763 - val_loss: 0.0538 - val_accuracy: 0.9838
Epoch 5/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0590 - accuracy: 0.9810 - val_loss: 0.0502 - val_accuracy: 0.9846
Epoch 6/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0531 - accuracy: 0.9824 - val_loss: 0.0517 - val_accuracy: 0.9839
Epoch 7/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0510 - accuracy: 0.9828 - val_loss: 0.0446 - val_accuracy: 0.9861
Epoch 8/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0412 - accuracy: 0.9871 - val_loss: 0.0442 - val_accuracy: 0.9866
Epoch 9/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0387 - accuracy: 0.9871 - val_loss: 0.0569 - val_accuracy: 0.9841
Epoch 10/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0373 - accuracy: 0.9879 - val_loss: 0.0451 - val_accuracy: 0.9862
Epoch 11/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0317 - accuracy: 0.9896 - val_loss: 0.0461 - val_accuracy: 0.9867
Epoch 12/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0286 - accuracy: 0.9906 - val_loss: 0.0524 - val_accuracy: 0.9858
Epoch 13/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0278 - accuracy: 0.9906 - val_loss: 0.0495 - val_accuracy: 0.9873
Epoch 14/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0238 - accuracy: 0.9918 - val_loss: 0.0557 - val_accuracy: 0.9849
Epoch 15/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0216 - accuracy: 0.9931 - val_loss: 0.0471 - val_accuracy: 0.9866
Epoch 16/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0214 - accuracy: 0.9924 - val_loss: 0.0454 - val_accuracy: 0.9880
Epoch 17/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0208 - accuracy: 0.9922 - val_loss: 0.0591 - val_accuracy: 0.9863
Epoch 18/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0196 - accuracy: 0.9928 - val_loss: 0.0518 - val_accuracy: 0.9875
Epoch 19/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0172 - accuracy: 0.9942 - val_loss: 0.0500 - val_accuracy: 0.9881
Epoch 20/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0171 - accuracy: 0.9940 - val_loss: 0.0420 - val_accuracy: 0.9890
Epoch 21/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0127 - accuracy: 0.9954 - val_loss: 0.0515 - val_accuracy: 0.9878
Epoch 22/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0141 - accuracy: 0.9952 - val_loss: 0.0459 - val_accuracy: 0.9879
Epoch 23/200
206/206 [==============================] - 2s 10ms/step - loss: 0.0150 - accuracy: 0.9950 - val_loss: 0.0474 - val_accuracy: 0.9882
Epoch 24/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0174 - accuracy: 0.9941 - val_loss: 0.0452 - val_accuracy: 0.9895
Epoch 25/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0164 - accuracy: 0.9939 - val_loss: 0.0452 - val_accuracy: 0.9893
Epoch 26/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0135 - accuracy: 0.9948 - val_loss: 0.0503 - val_accuracy: 0.9881
Epoch 27/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0145 - accuracy: 0.9955 - val_loss: 0.0429 - val_accuracy: 0.9902
Epoch 28/200
206/206 [==============================] - 2s 10ms/step - loss: 0.0122 - accuracy: 0.9963 - val_loss: 0.0490 - val_accuracy: 0.9896
Epoch 29/200
206/206 [==============================] - 2s 10ms/step - loss: 0.0097 - accuracy: 0.9968 - val_loss: 0.0512 - val_accuracy: 0.9897
Epoch 30/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0118 - accuracy: 0.9960 - val_loss: 0.0463 - val_accuracy: 0.9889
Epoch 31/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0091 - accuracy: 0.9967 - val_loss: 0.0523 - val_accuracy: 0.9888
Epoch 32/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0117 - accuracy: 0.9955 - val_loss: 0.0511 - val_accuracy: 0.9893
Epoch 33/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0123 - accuracy: 0.9958 - val_loss: 0.0490 - val_accuracy: 0.9883
Epoch 34/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0112 - accuracy: 0.9963 - val_loss: 0.0506 - val_accuracy: 0.9889
Epoch 35/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0084 - accuracy: 0.9975 - val_loss: 0.0561 - val_accuracy: 0.9873
Epoch 36/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0105 - accuracy: 0.9966 - val_loss: 0.0518 - val_accuracy: 0.9878
Epoch 37/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0076 - accuracy: 0.9971 - val_loss: 0.0496 - val_accuracy: 0.9889
Epoch 38/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0085 - accuracy: 0.9970 - val_loss: 0.0544 - val_accuracy: 0.9896
Epoch 39/200
206/206 [==============================] - 2s 10ms/step - loss: 0.0102 - accuracy: 0.9967 - val_loss: 0.0516 - val_accuracy: 0.9891
Epoch 40/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0085 - accuracy: 0.9968 - val_loss: 0.0490 - val_accuracy: 0.9887
Epoch 41/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0084 - accuracy: 0.9971 - val_loss: 0.0514 - val_accuracy: 0.9893
Epoch 42/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0100 - accuracy: 0.9962 - val_loss: 0.0518 - val_accuracy: 0.9884
Epoch 43/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0080 - accuracy: 0.9973 - val_loss: 0.0504 - val_accuracy: 0.9889
Epoch 44/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0082 - accuracy: 0.9970 - val_loss: 0.0524 - val_accuracy: 0.9890
Epoch 45/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0099 - accuracy: 0.9972 - val_loss: 0.0589 - val_accuracy: 0.9887
Epoch 46/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0078 - accuracy: 0.9981 - val_loss: 0.0526 - val_accuracy: 0.9885
Epoch 47/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0056 - accuracy: 0.9982 - val_loss: 0.0533 - val_accuracy: 0.9897
Epoch 48/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0066 - accuracy: 0.9977 - val_loss: 0.0486 - val_accuracy: 0.9890
Epoch 49/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0051 - accuracy: 0.9977 - val_loss: 0.0524 - val_accuracy: 0.9890
Epoch 50/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0071 - accuracy: 0.9977 - val_loss: 0.0610 - val_accuracy: 0.9882
Epoch 51/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0084 - accuracy: 0.9976 - val_loss: 0.0562 - val_accuracy: 0.9892
Epoch 52/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0089 - accuracy: 0.9965 - val_loss: 0.0487 - val_accuracy: 0.9906
Epoch 53/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0058 - accuracy: 0.9977 - val_loss: 0.0597 - val_accuracy: 0.9889
Epoch 54/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0094 - accuracy: 0.9966 - val_loss: 0.0561 - val_accuracy: 0.9899
Epoch 55/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0081 - accuracy: 0.9975 - val_loss: 0.0571 - val_accuracy: 0.9892
Epoch 56/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0059 - accuracy: 0.9979 - val_loss: 0.0570 - val_accuracy: 0.9890
Epoch 57/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0062 - accuracy: 0.9978 - val_loss: 0.0522 - val_accuracy: 0.9889
Epoch 58/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0067 - accuracy: 0.9975 - val_loss: 0.0568 - val_accuracy: 0.9878
Epoch 59/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0093 - accuracy: 0.9973 - val_loss: 0.0468 - val_accuracy: 0.9888
Epoch 60/200
206/206 [==============================] - 3s 13ms/step - loss: 0.0037 - accuracy: 0.9988 - val_loss: 0.0529 - val_accuracy: 0.9891
Epoch 61/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0077 - accuracy: 0.9970 - val_loss: 0.0570 - val_accuracy: 0.9880
Epoch 62/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0051 - accuracy: 0.9982 - val_loss: 0.0537 - val_accuracy: 0.9881
Epoch 63/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0063 - accuracy: 0.9980 - val_loss: 0.0685 - val_accuracy: 0.9875
Epoch 64/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0063 - accuracy: 0.9978 - val_loss: 0.0543 - val_accuracy: 0.9888
Epoch 65/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0053 - accuracy: 0.9983 - val_loss: 0.0578 - val_accuracy: 0.9879
Epoch 66/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0061 - accuracy: 0.9980 - val_loss: 0.0547 - val_accuracy: 0.9876
Epoch 67/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0072 - accuracy: 0.9979 - val_loss: 0.0618 - val_accuracy: 0.9880
Epoch 68/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0051 - accuracy: 0.9984 - val_loss: 0.0570 - val_accuracy: 0.9887
Epoch 69/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0042 - accuracy: 0.9987 - val_loss: 0.0609 - val_accuracy: 0.9887
Epoch 70/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0045 - accuracy: 0.9988 - val_loss: 0.0626 - val_accuracy: 0.9891
Epoch 71/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0050 - accuracy: 0.9983 - val_loss: 0.0690 - val_accuracy: 0.9895
Epoch 72/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0059 - accuracy: 0.9980 - val_loss: 0.0749 - val_accuracy: 0.9871
Epoch 73/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0050 - accuracy: 0.9983 - val_loss: 0.0741 - val_accuracy: 0.9883
Epoch 74/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0061 - accuracy: 0.9982 - val_loss: 0.0602 - val_accuracy: 0.9885
Epoch 75/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0047 - accuracy: 0.9984 - val_loss: 0.0649 - val_accuracy: 0.9873
Epoch 76/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0059 - accuracy: 0.9977 - val_loss: 0.0572 - val_accuracy: 0.9890
Epoch 77/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0040 - accuracy: 0.9986 - val_loss: 0.0697 - val_accuracy: 0.9880
Epoch 78/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0053 - accuracy: 0.9982 - val_loss: 0.0570 - val_accuracy: 0.9888
Epoch 79/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0056 - accuracy: 0.9981 - val_loss: 0.0662 - val_accuracy: 0.9898
Epoch 80/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0043 - accuracy: 0.9987 - val_loss: 0.0722 - val_accuracy: 0.9890
Epoch 81/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0057 - accuracy: 0.9983 - val_loss: 0.0583 - val_accuracy: 0.9900
Epoch 82/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0047 - accuracy: 0.9982 - val_loss: 0.0622 - val_accuracy: 0.9902
Epoch 83/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0049 - accuracy: 0.9982 - val_loss: 0.0655 - val_accuracy: 0.9895
Epoch 84/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0042 - accuracy: 0.9986 - val_loss: 0.0683 - val_accuracy: 0.9895
Epoch 85/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0068 - accuracy: 0.9978 - val_loss: 0.0639 - val_accuracy: 0.9888
Epoch 86/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0030 - accuracy: 0.9989 - val_loss: 0.0668 - val_accuracy: 0.9895
Epoch 87/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0031 - accuracy: 0.9990 - val_loss: 0.0687 - val_accuracy: 0.9895
Epoch 88/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0039 - accuracy: 0.9988 - val_loss: 0.0610 - val_accuracy: 0.9905
Epoch 89/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0042 - accuracy: 0.9984 - val_loss: 0.0577 - val_accuracy: 0.9900
Epoch 90/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0048 - accuracy: 0.9984 - val_loss: 0.0672 - val_accuracy: 0.9888
Epoch 91/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0054 - accuracy: 0.9981 - val_loss: 0.0645 - val_accuracy: 0.9899
Epoch 92/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0036 - accuracy: 0.9987 - val_loss: 0.0811 - val_accuracy: 0.9896
Epoch 93/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0062 - accuracy: 0.9983 - val_loss: 0.0628 - val_accuracy: 0.9888
Epoch 94/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0021 - accuracy: 0.9994 - val_loss: 0.0798 - val_accuracy: 0.9892
Epoch 95/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0041 - accuracy: 0.9985 - val_loss: 0.0739 - val_accuracy: 0.9874
Epoch 96/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0067 - accuracy: 0.9981 - val_loss: 0.0597 - val_accuracy: 0.9897
Epoch 97/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0053 - accuracy: 0.9984 - val_loss: 0.0595 - val_accuracy: 0.9895
Epoch 98/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0058 - accuracy: 0.9978 - val_loss: 0.0627 - val_accuracy: 0.9890
Epoch 99/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0050 - accuracy: 0.9984 - val_loss: 0.0688 - val_accuracy: 0.9891
Epoch 100/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0028 - accuracy: 0.9989 - val_loss: 0.0713 - val_accuracy: 0.9889
Epoch 101/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0046 - accuracy: 0.9987 - val_loss: 0.0580 - val_accuracy: 0.9900
Epoch 102/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0033 - accuracy: 0.9987 - val_loss: 0.0640 - val_accuracy: 0.9901
Epoch 103/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0023 - accuracy: 0.9994 - val_loss: 0.0693 - val_accuracy: 0.9895
Epoch 104/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0037 - accuracy: 0.9988 - val_loss: 0.0601 - val_accuracy: 0.9900
Epoch 105/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0054 - accuracy: 0.9983 - val_loss: 0.0652 - val_accuracy: 0.9896
Epoch 106/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0022 - accuracy: 0.9993 - val_loss: 0.0673 - val_accuracy: 0.9908
Epoch 107/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0051 - accuracy: 0.9985 - val_loss: 0.0693 - val_accuracy: 0.9889
Epoch 108/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0053 - accuracy: 0.9982 - val_loss: 0.0677 - val_accuracy: 0.9893
Epoch 109/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0037 - accuracy: 0.9990 - val_loss: 0.0703 - val_accuracy: 0.9895
Epoch 110/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0037 - accuracy: 0.9989 - val_loss: 0.0685 - val_accuracy: 0.9881
Epoch 111/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0048 - accuracy: 0.9984 - val_loss: 0.0659 - val_accuracy: 0.9895
Epoch 112/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0051 - accuracy: 0.9984 - val_loss: 0.0597 - val_accuracy: 0.9897
Epoch 113/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0053 - accuracy: 0.9983 - val_loss: 0.0762 - val_accuracy: 0.9895
Epoch 114/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0045 - accuracy: 0.9986 - val_loss: 0.0577 - val_accuracy: 0.9904
Epoch 115/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0025 - accuracy: 0.9990 - val_loss: 0.0616 - val_accuracy: 0.9893
Epoch 116/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0029 - accuracy: 0.9992 - val_loss: 0.0843 - val_accuracy: 0.9875
Epoch 117/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0058 - accuracy: 0.9982 - val_loss: 0.0644 - val_accuracy: 0.9896
Epoch 118/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0034 - accuracy: 0.9988 - val_loss: 0.0735 - val_accuracy: 0.9897
Epoch 119/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0043 - accuracy: 0.9986 - val_loss: 0.0676 - val_accuracy: 0.9890
Epoch 120/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0032 - accuracy: 0.9989 - val_loss: 0.0719 - val_accuracy: 0.9895
Epoch 121/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0018 - accuracy: 0.9994 - val_loss: 0.0764 - val_accuracy: 0.9880
Epoch 122/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0035 - accuracy: 0.9988 - val_loss: 0.0625 - val_accuracy: 0.9899
Epoch 123/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0046 - accuracy: 0.9985 - val_loss: 0.0703 - val_accuracy: 0.9892
Epoch 124/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0034 - accuracy: 0.9989 - val_loss: 0.0764 - val_accuracy: 0.9893
Epoch 125/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.0720 - val_accuracy: 0.9883
Epoch 126/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0038 - accuracy: 0.9989 - val_loss: 0.0752 - val_accuracy: 0.9895
Epoch 127/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0025 - accuracy: 0.9991 - val_loss: 0.0762 - val_accuracy: 0.9893
Epoch 128/200
206/206 [==============================] - 2s 12ms/step - loss: 0.0047 - accuracy: 0.9987 - val_loss: 0.0701 - val_accuracy: 0.9902
Epoch 129/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0032 - accuracy: 0.9989 - val_loss: 0.0806 - val_accuracy: 0.9882
Epoch 130/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0049 - accuracy: 0.9987 - val_loss: 0.0954 - val_accuracy: 0.9881
Epoch 131/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0064 - accuracy: 0.9980 - val_loss: 0.0666 - val_accuracy: 0.9898
Epoch 132/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0053 - accuracy: 0.9987 - val_loss: 0.0596 - val_accuracy: 0.9898
Epoch 133/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0030 - accuracy: 0.9992 - val_loss: 0.0646 - val_accuracy: 0.9910
Epoch 134/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0023 - accuracy: 0.9991 - val_loss: 0.0727 - val_accuracy: 0.9901
Epoch 135/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0052 - accuracy: 0.9984 - val_loss: 0.0778 - val_accuracy: 0.9893
Epoch 136/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0047 - accuracy: 0.9983 - val_loss: 0.0789 - val_accuracy: 0.9888
Epoch 137/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0043 - accuracy: 0.9988 - val_loss: 0.0788 - val_accuracy: 0.9895
Epoch 138/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0030 - accuracy: 0.9989 - val_loss: 0.0682 - val_accuracy: 0.9900
Epoch 139/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0015 - accuracy: 0.9996 - val_loss: 0.0731 - val_accuracy: 0.9899
Epoch 140/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0021 - accuracy: 0.9994 - val_loss: 0.0836 - val_accuracy: 0.9887
Epoch 141/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0052 - accuracy: 0.9987 - val_loss: 0.0815 - val_accuracy: 0.9888
Epoch 142/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0033 - accuracy: 0.9991 - val_loss: 0.0867 - val_accuracy: 0.9882
Epoch 143/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0036 - accuracy: 0.9989 - val_loss: 0.0796 - val_accuracy: 0.9888
Epoch 144/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0036 - accuracy: 0.9988 - val_loss: 0.0856 - val_accuracy: 0.9890
Epoch 145/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0037 - accuracy: 0.9989 - val_loss: 0.0788 - val_accuracy: 0.9893
Epoch 146/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0054 - accuracy: 0.9983 - val_loss: 0.0784 - val_accuracy: 0.9891
Epoch 147/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0036 - accuracy: 0.9991 - val_loss: 0.0836 - val_accuracy: 0.9887
Epoch 148/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0034 - accuracy: 0.9990 - val_loss: 0.0751 - val_accuracy: 0.9898
Epoch 149/200
206/206 [==============================] - 3s 13ms/step - loss: 0.0036 - accuracy: 0.9991 - val_loss: 0.0731 - val_accuracy: 0.9890
Epoch 150/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0056 - accuracy: 0.9984 - val_loss: 0.0663 - val_accuracy: 0.9899
Epoch 151/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0037 - accuracy: 0.9990 - val_loss: 0.0765 - val_accuracy: 0.9893
Epoch 152/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0027 - accuracy: 0.9992 - val_loss: 0.0725 - val_accuracy: 0.9890
Epoch 153/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0011 - accuracy: 0.9998 - val_loss: 0.0713 - val_accuracy: 0.9899
Epoch 154/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0012 - accuracy: 0.9994 - val_loss: 0.0896 - val_accuracy: 0.9884
Epoch 155/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0035 - accuracy: 0.9988 - val_loss: 0.0739 - val_accuracy: 0.9901
Epoch 156/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0052 - accuracy: 0.9987 - val_loss: 0.0754 - val_accuracy: 0.9889
Epoch 157/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0036 - accuracy: 0.9991 - val_loss: 0.0714 - val_accuracy: 0.9897
Epoch 158/200
206/206 [==============================] - 3s 13ms/step - loss: 0.0018 - accuracy: 0.9992 - val_loss: 0.0822 - val_accuracy: 0.9891
Epoch 159/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0028 - accuracy: 0.9992 - val_loss: 0.0703 - val_accuracy: 0.9910
Epoch 160/200
206/206 [==============================] - 3s 13ms/step - loss: 0.0039 - accuracy: 0.9988 - val_loss: 0.0735 - val_accuracy: 0.9899
Epoch 161/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0063 - accuracy: 0.9983 - val_loss: 0.0945 - val_accuracy: 0.9890
Epoch 162/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0030 - accuracy: 0.9992 - val_loss: 0.0742 - val_accuracy: 0.9890
Epoch 163/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0026 - accuracy: 0.9991 - val_loss: 0.0800 - val_accuracy: 0.9897
Epoch 164/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0032 - accuracy: 0.9991 - val_loss: 0.0777 - val_accuracy: 0.9895
Epoch 165/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0041 - accuracy: 0.9983 - val_loss: 0.0904 - val_accuracy: 0.9891
Epoch 166/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0059 - accuracy: 0.9985 - val_loss: 0.0772 - val_accuracy: 0.9906
Epoch 167/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0049 - accuracy: 0.9988 - val_loss: 0.0775 - val_accuracy: 0.9888
Epoch 168/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0021 - accuracy: 0.9993 - val_loss: 0.0791 - val_accuracy: 0.9898
Epoch 169/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0029 - accuracy: 0.9993 - val_loss: 0.0689 - val_accuracy: 0.9902
Epoch 170/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0020 - accuracy: 0.9994 - val_loss: 0.0767 - val_accuracy: 0.9892
Epoch 171/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0043 - accuracy: 0.9990 - val_loss: 0.0720 - val_accuracy: 0.9897
Epoch 172/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0027 - accuracy: 0.9992 - val_loss: 0.0707 - val_accuracy: 0.9892
Epoch 173/200
206/206 [==============================] - 3s 13ms/step - loss: 0.0041 - accuracy: 0.9988 - val_loss: 0.0730 - val_accuracy: 0.9895
Epoch 174/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0035 - accuracy: 0.9990 - val_loss: 0.0782 - val_accuracy: 0.9891
Epoch 175/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0058 - accuracy: 0.9983 - val_loss: 0.0796 - val_accuracy: 0.9891
Epoch 176/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0029 - accuracy: 0.9991 - val_loss: 0.0890 - val_accuracy: 0.9875
Epoch 177/200
206/206 [==============================] - 3s 13ms/step - loss: 0.0015 - accuracy: 0.9995 - val_loss: 0.0772 - val_accuracy: 0.9895
Epoch 178/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0022 - accuracy: 0.9992 - val_loss: 0.0828 - val_accuracy: 0.9884
Epoch 179/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0023 - accuracy: 0.9992 - val_loss: 0.0721 - val_accuracy: 0.9898
Epoch 180/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0024 - accuracy: 0.9992 - val_loss: 0.0836 - val_accuracy: 0.9887
Epoch 181/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0033 - accuracy: 0.9991 - val_loss: 0.0798 - val_accuracy: 0.9902
Epoch 182/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0028 - accuracy: 0.9991 - val_loss: 0.0742 - val_accuracy: 0.9895
Epoch 183/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0023 - accuracy: 0.9994 - val_loss: 0.0969 - val_accuracy: 0.9883
Epoch 184/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0040 - accuracy: 0.9992 - val_loss: 0.0783 - val_accuracy: 0.9889
Epoch 185/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0029 - accuracy: 0.9990 - val_loss: 0.0831 - val_accuracy: 0.9887
Epoch 186/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0027 - accuracy: 0.9993 - val_loss: 0.0721 - val_accuracy: 0.9893
Epoch 187/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0040 - accuracy: 0.9989 - val_loss: 0.0802 - val_accuracy: 0.9884
Epoch 188/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0041 - accuracy: 0.9986 - val_loss: 0.0762 - val_accuracy: 0.9892
Epoch 189/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0043 - accuracy: 0.9987 - val_loss: 0.0774 - val_accuracy: 0.9890
Epoch 190/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0029 - accuracy: 0.9990 - val_loss: 0.0880 - val_accuracy: 0.9889
Epoch 191/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0016 - accuracy: 0.9995 - val_loss: 0.0805 - val_accuracy: 0.9898
Epoch 192/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0036 - accuracy: 0.9990 - val_loss: 0.0927 - val_accuracy: 0.9871
Epoch 193/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0081 - accuracy: 0.9979 - val_loss: 0.1066 - val_accuracy: 0.9865
Epoch 194/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0044 - accuracy: 0.9985 - val_loss: 0.1070 - val_accuracy: 0.9874
Epoch 195/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0034 - accuracy: 0.9992 - val_loss: 0.0783 - val_accuracy: 0.9892
Epoch 196/200
206/206 [==============================] - 3s 13ms/step - loss: 0.0049 - accuracy: 0.9990 - val_loss: 0.0807 - val_accuracy: 0.9884
Epoch 197/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0014 - accuracy: 0.9995 - val_loss: 0.0744 - val_accuracy: 0.9899
Epoch 198/200
206/206 [==============================] - 3s 12ms/step - loss: 3.8746e-04 - accuracy: 0.9999 - val_loss: 0.0744 - val_accuracy: 0.9893
Epoch 199/200
206/206 [==============================] - 3s 12ms/step - loss: 0.0016 - accuracy: 0.9997 - val_loss: 0.0840 - val_accuracy: 0.9897
Epoch 200/200
206/206 [==============================] - 2s 11ms/step - loss: 0.0033 - accuracy: 0.9990 - val_loss: 0.0877 - val_accuracy: 0.9892
</pre>

```python
print(history)  # History 객체
```

```python
<keras.callbacks.History object at 0x7f7ae02f1d10>
```

```python
print(type(history))
print(history.history)
print(history.history.keys())
```

```python
<class 'keras.callbacks.History'>
{'loss': [0.004623409826308489, 0.006843957118690014, 0.0032009996939450502, 0.0045742252841591835, 0.004606267903000116, 0.006238088943064213, 0.004245389718562365, 0.003613620065152645, 0.0050657931715250015, 0.006343253422528505, 0.007855178788304329, 0.006581406574696302, 0.005084768403321505, 0.004552390892058611, 0.0032064232509583235, 0.00605241023004055, 0.006683922838419676, 0.0029638411942869425, 0.004308552481234074, 0.003379883710294962, 0.004657298792153597, 0.004567190073430538, 0.0024137182626873255, 0.003454821649938822, 0.0015327066648751497, 0.006218715105205774, 0.007088629063218832, 0.0031871746759861708, 0.003786715678870678, 0.004721163306385279, 0.004570700228214264, 0.00602500606328249, 0.005826263688504696, 0.006561106536537409, 0.002542713889852166, 0.00230606272816658, 0.004497144371271133, 0.007553324569016695, 0.003829097142443061, 0.002783156232908368, 0.0028674935456365347, 0.0031703978311270475, 0.0037965464871376753, 0.004146551713347435, 0.004800420254468918, 0.006363581400364637, 0.004784253891557455, 0.004718154203146696, 0.002734947018325329, 0.004815686959773302, 0.004429407883435488, 0.003562333295121789, 0.0037042596377432346, 0.004566493444144726, 0.0037548106629401445, 0.003991194535046816, 0.0022207906004041433, 0.0013118961360305548, 0.002465906785801053, 0.004831644240766764, 0.0034821382723748684, 0.004210793878883123, 0.0048467619344592094, 0.0045460923574864864, 0.00432539964094758, 0.0037853100802749395, 0.0024083892349153757, 0.003982254303991795, 0.003188611939549446, 0.003525683656334877, 0.005808182992041111, 0.005727812647819519, 0.0025379746221005917, 0.0034094403963536024, 0.004197009839117527, 0.0020138388499617577, 0.002019082196056843, 0.004737646318972111, 0.004107393324375153, 0.002260342240333557, 0.0035998905077576637, 0.00418312381953001, 0.003988245967775583, 0.005659411661326885, 0.0038970906753093004, 0.0025869214441627264, 0.0038371323607861996, 0.004869896452873945, 0.0017317667370662093, 0.0035642010625451803, 0.003181649837642908, 0.0037993276491761208, 0.0034856030251830816, 0.004315475467592478, 0.0038580994587391615, 0.005171475000679493, 0.0034616657067090273, 0.002118348143994808, 0.0010973454918712378, 0.002196169225499034, 0.003876000875607133, 0.004473382607102394, 0.004866133909672499, 0.003330022329464555, 0.003979191649705172, 0.0038439854979515076, 0.0030118569266051054, 0.003138019936159253, 0.003941423259675503, 0.0024744824040681124, 0.0028882171027362347, 0.0027873932849615812, 0.004368187859654427, 0.0028511430136859417, 0.0011392546584829688, 0.0033697118051350117, 0.0018793778726831079, 0.004919940140098333, 0.004307009279727936, 0.004824570845812559, 0.003966158721596003, 0.0029201938305050135, 0.002379107056185603, 0.003476317971944809, 0.002174708293750882, 0.002976282499730587, 0.002722567180171609, 0.0028261656407266855, 0.002839328721165657, 0.0025814592372626066, 0.004874161910265684, 0.006018346641212702, 0.001614722074009478, 0.002353058662265539, 0.0011378986528143287, 0.002241802169010043, 0.0019329024944454432, 0.0019673171918839216, 0.004685606341809034, 0.003239949932321906, 0.0022494245786219835, 0.005073073785752058, 0.005633389111608267, 0.0022417104337364435, 0.0030067129991948605, 0.0037173291202634573, 0.0029204401653259993, 0.003711042460054159, 0.00258720968849957, 0.0007485477835871279, 0.003200979670509696, 0.002314990386366844, 0.0032880217768251896, 0.0025198881048709154, 0.0024552049580961466, 0.0022530220448970795, 0.0007356816786341369, 0.0016705808229744434, 0.0038414401933550835, 0.006209112703800201, 0.0027432446368038654, 0.0019680580589920282, 0.0021855419036000967, 0.004228970967233181, 0.001504129613749683, 0.0022565892431885004, 0.0028447320219129324, 0.006767620798200369, 0.002589187119156122, 0.0014656755374744534, 0.003003062680363655, 0.001219979370944202, 0.0016279941191896796, 0.0038380399346351624, 0.001350936945527792, 0.002709262305870652, 0.00404229573905468, 0.0046625505201518536, 0.0023730110842734575, 0.0026305268984287977, 0.002389748813584447, 0.0025454291608184576, 0.00211035693064332, 0.0036345133557915688, 0.006584291812032461, 0.0032832343131303787, 0.0018566123908385634, 0.0013342718593776226, 0.0020881732925772667, 0.0023678611032664776, 0.0037429824005812407, 0.0027418676763772964, 0.004012516234070063, 0.0011757820611819625, 0.0013707428006455302, 0.0034502933267503977, 0.001087260665372014, 0.0022618877701461315, 0.0068222032859921455, 0.0039827884174883366], 'accuracy': [0.9986880421638489, 0.9978619813919067, 0.9989795684814453, 0.9982993006706238, 0.998590886592865, 0.9984450936317444, 0.9983965158462524, 0.9988338351249695, 0.9982993006706238, 0.9981049299240112, 0.9976190328598022, 0.9977162480354309, 0.9983479380607605, 0.9986394643783569, 0.9989795684814453, 0.9979591965675354, 0.9979106187820435, 0.9987852573394775, 0.9982993006706238, 0.9986880421638489, 0.9982507228851318, 0.9987852573394775, 0.9989795684814453, 0.999028205871582, 0.999514102935791, 0.9977162480354309, 0.9978619813919067, 0.9989309906959534, 0.9986394643783569, 0.9983479380607605, 0.9984450936317444, 0.9982507228851318, 0.9984936714172363, 0.9978619813919067, 0.999028205871582, 0.999076783657074, 0.9987366199493408, 0.9979106187820435, 0.9987366199493408, 0.9992711544036865, 0.9991253614425659, 0.9994168877601624, 0.9986394643783569, 0.999028205871582, 0.9986880421638489, 0.9983479380607605, 0.9989309906959534, 0.9987852573394775, 0.9989309906959534, 0.9984450936317444, 0.9989309906959534, 0.9988338351249695, 0.9989309906959534, 0.9985422492027283, 0.9986880421638489, 0.9988338351249695, 0.9992711544036865, 0.9993683099746704, 0.9992225170135498, 0.9984936714172363, 0.9987852573394775, 0.9984936714172363, 0.9986394643783569, 0.9987852573394775, 0.9983479380607605, 0.9986880421638489, 0.9992711544036865, 0.9987366199493408, 0.999028205871582, 0.9989795684814453, 0.9983479380607605, 0.9984936714172363, 0.9992225170135498, 0.999076783657074, 0.9988338351249695, 0.9989309906959534, 0.9993683099746704, 0.9988824129104614, 0.9988824129104614, 0.999028205871582, 0.9988338351249695, 0.999076783657074, 0.9989795684814453, 0.9984936714172363, 0.9988338351249695, 0.999514102935791, 0.9989309906959534, 0.9987366199493408, 0.9996112585067749, 0.9989309906959534, 0.999076783657074, 0.9987366199493408, 0.9991253614425659, 0.9987852573394775, 0.9989795684814453, 0.998590886592865, 0.9987852573394775, 0.9992225170135498, 0.9997084736824036, 0.9993197321891785, 0.9989795684814453, 0.9986880421638489, 0.9988338351249695, 0.9992711544036865, 0.9989795684814453, 0.999028205871582, 0.999028205871582, 0.9988338351249695, 0.9989309906959534, 0.9993197321891785, 0.9991253614425659, 0.9987366199493408, 0.9989309906959534, 0.9989795684814453, 0.9997084736824036, 0.999076783657074, 0.9994655251502991, 0.9982993006706238, 0.9989309906959534, 0.9989309906959534, 0.998590886592865, 0.999028205871582, 0.9992711544036865, 0.999028205871582, 0.9994168877601624, 0.999076783657074, 0.999076783657074, 0.9993197321891785, 0.9991253614425659, 0.9992711544036865, 0.9984936714172363, 0.9986880421638489, 0.999562680721283, 0.9992225170135498, 0.999562680721283, 0.9992711544036865, 0.999514102935791, 0.9994655251502991, 0.9985422492027283, 0.9989309906959534, 0.9994168877601624, 0.9984450936317444, 0.9988338351249695, 0.9994168877601624, 0.9989309906959534, 0.9991253614425659, 0.9993683099746704, 0.999076783657074, 0.9991253614425659, 0.9998542070388794, 0.9992711544036865, 0.9994655251502991, 0.9992711544036865, 0.9989795684814453, 0.999076783657074, 0.9993197321891785, 0.9997084736824036, 0.9994655251502991, 0.999076783657074, 0.9987366199493408, 0.9991253614425659, 0.9993197321891785, 0.9993683099746704, 0.9987366199493408, 0.9996598362922668, 0.9993683099746704, 0.9991739392280579, 0.9985422492027283, 0.9992225170135498, 0.9994655251502991, 0.9991253614425659, 0.9997570514678955, 0.999562680721283, 0.999028205871582, 0.9994655251502991, 0.999514102935791, 0.9991253614425659, 0.9987852573394775, 0.9992711544036865, 0.9991253614425659, 0.9992225170135498, 0.9993197321891785, 0.999562680721283, 0.9991253614425659, 0.9986880421638489, 0.9992225170135498, 0.9993197321891785, 0.9996112585067749, 0.9993683099746704, 0.9994168877601624, 0.9991253614425659, 0.9989309906959534, 0.9988338351249695, 0.999562680721283, 0.999514102935791, 0.9993197321891785, 0.9997570514678955, 0.9993197321891785, 0.9984450936317444, 0.9989795684814453], 'val_loss': [0.0812021791934967, 0.055882617831230164, 0.0559806153178215, 0.06365993618965149, 0.061592064797878265, 0.05684076249599457, 0.05699678510427475, 0.05988270044326782, 0.06428500264883041, 0.06533295661211014, 0.05684195086359978, 0.06627634167671204, 0.05885637551546097, 0.0612090565264225, 0.06447108089923859, 0.0648312121629715, 0.06858304888010025, 0.06142117828130722, 0.06105095520615578, 0.06188955903053284, 0.058921195566654205, 0.05485398694872856, 0.05367336794734001, 0.06132013350725174, 0.06951411813497543, 0.07575992494821548, 0.06695005297660828, 0.08225492388010025, 0.0931745395064354, 0.06370662897825241, 0.08157482743263245, 0.07068793475627899, 0.06098853051662445, 0.06294286996126175, 0.06718671321868896, 0.08080042153596878, 0.08415208011865616, 0.06737890094518661, 0.07294299453496933, 0.06388671696186066, 0.06858135014772415, 0.06913936883211136, 0.07432481646537781, 0.06405046582221985, 0.06636065244674683, 0.06817314028739929, 0.06461665034294128, 0.058683574199676514, 0.063783198595047, 0.07374031841754913, 0.06308551877737045, 0.06403356045484543, 0.06518860161304474, 0.06553389132022858, 0.06535477191209793, 0.06662313640117645, 0.06218307465314865, 0.06282678991556168, 0.05819014832377434, 0.06661631911993027, 0.08780357986688614, 0.07441122829914093, 0.07127929478883743, 0.06207586079835892, 0.07244846969842911, 0.0700213760137558, 0.06844360381364822, 0.07191403955221176, 0.07422049343585968, 0.08103184401988983, 0.07490262389183044, 0.06157075613737106, 0.07503896206617355, 0.0773785263299942, 0.07120078802108765, 0.07352291792631149, 0.08779837191104889, 0.07864716649055481, 0.07240544259548187, 0.0668560341000557, 0.06674767285585403, 0.06419073790311813, 0.08562057465314865, 0.05910971760749817, 0.062003884464502335, 0.08018448203802109, 0.06927365064620972, 0.08354604989290237, 0.07288294285535812, 0.07081222534179688, 0.06666053086519241, 0.0781710222363472, 0.08029373735189438, 0.06880756467580795, 0.08442626148462296, 0.10881539434194565, 0.07850266993045807, 0.07801799476146698, 0.06859110295772552, 0.08103664219379425, 0.09017189592123032, 0.08123304694890976, 0.09295051544904709, 0.06733733415603638, 0.06914892792701721, 0.06800424307584763, 0.0829770416021347, 0.07865701615810394, 0.06074867025017738, 0.07846616953611374, 0.07518523931503296, 0.08979342877864838, 0.06557976454496384, 0.07107502222061157, 0.07728635519742966, 0.06766694039106369, 0.0746350809931755, 0.09095939993858337, 0.08230981230735779, 0.06997477263212204, 0.09243179112672806, 0.07890462130308151, 0.09119124710559845, 0.08329521119594574, 0.0703558549284935, 0.0677255243062973, 0.08620481938123703, 0.0880238488316536, 0.0813581719994545, 0.08371482789516449, 0.0921851322054863, 0.074732206761837, 0.08753352612257004, 0.08009880036115646, 0.09931177645921707, 0.09833075851202011, 0.08807888627052307, 0.11088809370994568, 0.10106971114873886, 0.09894579648971558, 0.09414488077163696, 0.14184032380580902, 0.08533745259046555, 0.0873890370130539, 0.10895585268735886, 0.09809763729572296, 0.08469302207231522, 0.08591552823781967, 0.07613881677389145, 0.07676329463720322, 0.08455076813697815, 0.08109411597251892, 0.07695656269788742, 0.0850965678691864, 0.0750415250658989, 0.07913553714752197, 0.0971216931939125, 0.0810384601354599, 0.09570248425006866, 0.08461637794971466, 0.10137956589460373, 0.09228765964508057, 0.09951479732990265, 0.07685574144124985, 0.10058075189590454, 0.08597622066736221, 0.0994650200009346, 0.0908738300204277, 0.08670838922262192, 0.10386177152395248, 0.08663545548915863, 0.09306973963975906, 0.10351810604333878, 0.09868231415748596, 0.09461776167154312, 0.0902668684720993, 0.09699171036481857, 0.09531603008508682, 0.10014007240533829, 0.11003240942955017, 0.11466831713914871, 0.09762189537286758, 0.10286286473274231, 0.10803081840276718, 0.08512455224990845, 0.1001250222325325, 0.09475059062242508, 0.0927470475435257, 0.12587036192417145, 0.11294429004192352, 0.0967407152056694, 0.11819600313901901, 0.11461371183395386, 0.11303912103176117, 0.11757408082485199, 0.1358254849910736, 0.11665847152471542, 0.11307429522275925, 0.1156618744134903, 0.09618387371301651], 'val_accuracy': [0.988095223903656, 0.9893423914909363, 0.9908163547515869, 0.9900226593017578, 0.9891156554222107, 0.9896825551986694, 0.9883220195770264, 0.9895691871643066, 0.9892290234565735, 0.9887754917144775, 0.9887754917144775, 0.9884353876113892, 0.9897959232330322, 0.9893423914909363, 0.9896825551986694, 0.9879818558692932, 0.9888888597488403, 0.9902494549751282, 0.991723358631134, 0.9901360273361206, 0.9907029271125793, 0.9920634627342224, 0.9909297227859497, 0.990362823009491, 0.9901360273361206, 0.9884353876113892, 0.9888888597488403, 0.9890022873878479, 0.9895691871643066, 0.9904761910438538, 0.9879818558692932, 0.9878684878349304, 0.9901360273361206, 0.9893423914909363, 0.9901360273361206, 0.9895691871643066, 0.9886621236801147, 0.9900226593017578, 0.9900226593017578, 0.9911564588546753, 0.989909291267395, 0.9905895590782166, 0.9901360273361206, 0.9897959232330322, 0.989909291267395, 0.9897959232330322, 0.9908163547515869, 0.9914966225624084, 0.9909297227859497, 0.988548755645752, 0.9888888597488403, 0.9907029271125793, 0.9911564588546753, 0.9904761910438538, 0.9901360273361206, 0.9896825551986694, 0.990362823009491, 0.9913831949234009, 0.991723358631134, 0.9908163547515869, 0.9891156554222107, 0.990362823009491, 0.9895691871643066, 0.9912698268890381, 0.9900226593017578, 0.9893423914909363, 0.990362823009491, 0.9904761910438538, 0.9897959232330322, 0.9897959232330322, 0.990362823009491, 0.9909297227859497, 0.9894557595252991, 0.989909291267395, 0.990362823009491, 0.990362823009491, 0.9892290234565735, 0.9894557595252991, 0.9909297227859497, 0.9908163547515869, 0.9907029271125793, 0.9910430908203125, 0.9890022873878479, 0.9905895590782166, 0.9905895590782166, 0.9900226593017578, 0.9910430908203125, 0.990362823009491, 0.9902494549751282, 0.9908163547515869, 0.9909297227859497, 0.9911564588546753, 0.9893423914909363, 0.9902494549751282, 0.9896825551986694, 0.9870748519897461, 0.9904761910438538, 0.9897959232330322, 0.990362823009491, 0.9901360273361206, 0.9896825551986694, 0.9907029271125793, 0.9895691871643066, 0.9904761910438538, 0.9896825551986694, 0.9908163547515869, 0.9900226593017578, 0.9908163547515869, 0.9907029271125793, 0.9895691871643066, 0.9895691871643066, 0.9911564588546753, 0.9911564588546753, 0.9908163547515869, 0.9905895590782166, 0.9913831949234009, 0.9909297227859497, 0.9897959232330322, 0.9905895590782166, 0.990362823009491, 0.989909291267395, 0.9905895590782166, 0.9890022873878479, 0.9901360273361206, 0.9909297227859497, 0.9908163547515869, 0.9908163547515869, 0.9892290234565735, 0.9908163547515869, 0.9902494549751282, 0.9900226593017578, 0.9908163547515869, 0.9904761910438538, 0.9901360273361206, 0.9892290234565735, 0.9896825551986694, 0.9909297227859497, 0.9900226593017578, 0.9883220195770264, 0.990362823009491, 0.9905895590782166, 0.9859410524368286, 0.9910430908203125, 0.9904761910438538, 0.9887754917144775, 0.989909291267395, 0.9900226593017578, 0.9902494549751282, 0.9909297227859497, 0.9911564588546753, 0.9908163547515869, 0.9897959232330322, 0.9911564588546753, 0.990362823009491, 0.9925169944763184, 0.9911564588546753, 0.9907029271125793, 0.9904761910438538, 0.9897959232330322, 0.9904761910438538, 0.9890022873878479, 0.9907029271125793, 0.9890022873878479, 0.9901360273361206, 0.9901360273361206, 0.9908163547515869, 0.9896825551986694, 0.9902494549751282, 0.9905895590782166, 0.9888888597488403, 0.9904761910438538, 0.9914966225624084, 0.9900226593017578, 0.9912698268890381, 0.9910430908203125, 0.9905895590782166, 0.9907029271125793, 0.9902494549751282, 0.988548755645752, 0.988548755645752, 0.9904761910438538, 0.9904761910438538, 0.9901360273361206, 0.9890022873878479, 0.9901360273361206, 0.9893423914909363, 0.9919500946998596, 0.9902494549751282, 0.9888888597488403, 0.990362823009491, 0.990362823009491, 0.9886621236801147, 0.9894557595252991, 0.9904761910438538, 0.990362823009491, 0.988095223903656, 0.9883220195770264, 0.9892290234565735, 0.9902494549751282, 0.9901360273361206]}
dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
```
## 학습된 model 저장하기

- 모델의 구조 + 계산된 모든 weight, bias를 하나의 파일에 저장

- 확장자는 .h5 (HDF5) 형식



```python
model.save('/content/drive/MyDrive/colab/mnist_model_save/my_mnist_model.h5')
```

## model을 이용한 evaluation(평가)



```python
model.evaluate(norm_test_x_data.reshape(-1, 28, 28, 1), test_t_data)
#        loss             accuracy
```

```python
394/394 [==============================] - 2s 5ms/step - loss: 0.1422 - accuracy: 0.9914


[0.14221715927124023, 0.991428554058075]
```
## 저장된 모델을 불러와서 성능평가 진행하기


### 1. load_model



```python
from tensorflow.keras.models import load_model

new_model = load_model('/content/drive/MyDrive/colab/mnist_model_save/my_mnist_model.h5')

new_model.evaluate(norm_test_x_data.reshape(-1, 28, 28, 1), test_t_data)
#        loss             accuracy
```

```python
394/394 [==============================] - 5s 6ms/step - loss: 0.1422 - accuracy: 0.9914


[0.14221715927124023, 0.991428554058075]
```
### 2. chkpt, earlyStopping callback 포함



```python
# model 학습
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# checkpoint 설정
checkpoint_path = '/content/drive/MyDrive/colab/mnist_model_save/cp-{epoch:04d}.ckpt'
cp_callback = ModelCheckpoint(checkpoint_path,
                              save_weights_only=True,
                              period=5,  # 5 epoch마다 저장
                              verbose=1)

# earlyStopping
es = EarlyStopping(monitor='val_loss',
                   min_delta=0.001,  # 생략 시 - 값이 떨어지면 유효한 것으로 판단
                   patience=5,
                   verbose=1,
                   mode='auto',
                   restore_best_weights=True)

history = model.fit(norm_train_x_data.reshape(-1, 28, 28, 1),
                    train_t_data,
                    epochs=50,
                    batch_size=100,
                    verbose=1,
                    validation_split=0.3,
                    callbacks=[cp_callback, es])
```

<pre>
Epoch 1/50
206/206 [==============================] - 4s 18ms/step - loss: 0.0051 - accuracy: 0.9987 - val_loss: 0.0841 - val_accuracy: 0.9888
Epoch 2/50
206/206 [==============================] - 3s 17ms/step - loss: 0.0030 - accuracy: 0.9991 - val_loss: 0.0896 - val_accuracy: 0.9892
Epoch 3/50
206/206 [==============================] - 3s 13ms/step - loss: 0.0042 - accuracy: 0.9989 - val_loss: 0.0874 - val_accuracy: 0.9887
Epoch 4/50
206/206 [==============================] - 3s 12ms/step - loss: 0.0024 - accuracy: 0.9992 - val_loss: 0.0925 - val_accuracy: 0.9892
Epoch 5/50
205/206 [============================>.] - ETA: 0s - loss: 0.0022 - accuracy: 0.9994
Epoch 5: saving model to /content/drive/MyDrive/colab/mnist_model_save/cp-0005.ckpt
206/206 [==============================] - 3s 14ms/step - loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.0757 - val_accuracy: 0.9898
Epoch 6/50
206/206 [==============================] - 2s 11ms/step - loss: 0.0028 - accuracy: 0.9993 - val_loss: 0.0987 - val_accuracy: 0.9882
Epoch 7/50
206/206 [==============================] - 2s 12ms/step - loss: 0.0031 - accuracy: 0.9990 - val_loss: 0.0886 - val_accuracy: 0.9890
Epoch 8/50
206/206 [==============================] - 3s 12ms/step - loss: 0.0034 - accuracy: 0.9989 - val_loss: 0.0854 - val_accuracy: 0.9895
Epoch 9/50
206/206 [==============================] - 2s 11ms/step - loss: 0.0046 - accuracy: 0.9988 - val_loss: 0.0813 - val_accuracy: 0.9904
Epoch 10/50
206/206 [==============================] - ETA: 0s - loss: 0.0032 - accuracy: 0.9990
Epoch 10: saving model to /content/drive/MyDrive/colab/mnist_model_save/cp-0010.ckpt
Restoring model weights from the end of the best epoch: 5.
206/206 [==============================] - 3s 14ms/step - loss: 0.0032 - accuracy: 0.9990 - val_loss: 0.0787 - val_accuracy: 0.9902
Epoch 10: early stopping
</pre>
