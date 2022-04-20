# Cat_Dog data로 CNN 구현하기 🐱🐶


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```


```python
# Raw Data Loading
df = pd.read_csv('./data/kaggle/cat_dog/train.csv')
display(df.head())
print(df.shape)
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
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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


    (25000, 6401)
    


```python
x_data = df.drop('label', axis=1, inplace=False).values
t_data = df['label'].values
```


```python
plt.imshow(x_data[152:153].reshape(80, 80), cmap='gray')
plt.show()
```


    
![png](/Machine-Learning/images/0418/output_4_0.png)
    



```python
# data split
train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(x_data,
                 t_data,
                 test_size=0.3,
                 stratify=t_data)

# Normalization
scaler = MinMaxScaler()
scaler.fit(train_x_data)

norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)
```


```python
# model 생성
model = Sequential()

# CNN - Feature Extraction
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='SAME',
                 input_shape=(80, 80, 1)))  # 흑백이므로 channel 수는 1

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='SAME'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu',
                 padding='SAME'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# FC Layer
model.add(Flatten())
model.add(Dropout(rate=0.5))

# Hidden Layer
model.add(Dense(units=256,
                activation='relu'))

model.add(Dense(units=1,
                activation='sigmoid'))

print(model.summary())
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 80, 80, 64)        640       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 40, 40, 64)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 40, 40, 128)       73856     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 20, 20, 128)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 20, 20, 64)        73792     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 10, 10, 64)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 6400)              0         
    _________________________________________________________________
    dropout (Dropout)            (None, 6400)              0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               1638656   
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 1,787,201
    Trainable params: 1,787,201
    Non-trainable params: 0
    _________________________________________________________________
    None
    


```python
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```


```python
from timeit import default_timer as timer
from datetime import timedelta
start = timer()

history = model.fit(norm_train_x_data.reshape(-1, 80, 80, 1),
                    train_t_data.reshape(-1, 1),
                    epochs=200,
                    batch_size=100,
                    verbose=1,
                    validation_split=0.3)

end = timer()
print(f'Time elapsed : {timedelta(seconds=end-start)}')
```

    Epoch 1/200
    123/123 [==============================] - 8s 40ms/step - loss: 0.6776 - accuracy: 0.5552 - val_loss: 0.6561 - val_accuracy: 0.6046
    Epoch 2/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.6419 - accuracy: 0.6206 - val_loss: 0.6274 - val_accuracy: 0.6400
    Epoch 3/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.6141 - accuracy: 0.6554 - val_loss: 0.6080 - val_accuracy: 0.6691
    Epoch 4/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.5859 - accuracy: 0.6873 - val_loss: 0.5869 - val_accuracy: 0.6851
    Epoch 5/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.5640 - accuracy: 0.7060 - val_loss: 0.5637 - val_accuracy: 0.7093
    Epoch 6/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.5512 - accuracy: 0.7159 - val_loss: 0.5564 - val_accuracy: 0.7160
    Epoch 7/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.5390 - accuracy: 0.7290 - val_loss: 0.5736 - val_accuracy: 0.6958
    Epoch 8/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.5292 - accuracy: 0.7339 - val_loss: 0.5544 - val_accuracy: 0.7181
    Epoch 9/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.5151 - accuracy: 0.7470 - val_loss: 0.5370 - val_accuracy: 0.7270
    Epoch 10/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.5056 - accuracy: 0.7546 - val_loss: 0.5288 - val_accuracy: 0.7328
    Epoch 11/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.4993 - accuracy: 0.7536 - val_loss: 0.5314 - val_accuracy: 0.7326
    Epoch 12/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4894 - accuracy: 0.7607 - val_loss: 0.5303 - val_accuracy: 0.7375
    Epoch 13/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4802 - accuracy: 0.7655 - val_loss: 0.5170 - val_accuracy: 0.7448
    Epoch 14/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4750 - accuracy: 0.7722 - val_loss: 0.5195 - val_accuracy: 0.7408
    Epoch 15/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4697 - accuracy: 0.7755 - val_loss: 0.5193 - val_accuracy: 0.7438
    Epoch 16/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4631 - accuracy: 0.7769 - val_loss: 0.5149 - val_accuracy: 0.7476
    Epoch 17/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4567 - accuracy: 0.7834 - val_loss: 0.5152 - val_accuracy: 0.7499
    Epoch 18/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4446 - accuracy: 0.7904 - val_loss: 0.5091 - val_accuracy: 0.7535
    Epoch 19/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4427 - accuracy: 0.7918 - val_loss: 0.5097 - val_accuracy: 0.7512
    Epoch 20/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4324 - accuracy: 0.7991 - val_loss: 0.5136 - val_accuracy: 0.7465
    Epoch 21/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4305 - accuracy: 0.7996 - val_loss: 0.5298 - val_accuracy: 0.7400
    Epoch 22/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4160 - accuracy: 0.8061 - val_loss: 0.5069 - val_accuracy: 0.7528
    Epoch 23/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4129 - accuracy: 0.8094 - val_loss: 0.5115 - val_accuracy: 0.7522
    Epoch 24/200
    123/123 [==============================] - 4s 34ms/step - loss: 0.4079 - accuracy: 0.8088 - val_loss: 0.5099 - val_accuracy: 0.7535
    Epoch 25/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.3976 - accuracy: 0.8191 - val_loss: 0.5155 - val_accuracy: 0.7516
    Epoch 26/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.3990 - accuracy: 0.8198 - val_loss: 0.5087 - val_accuracy: 0.7503
    Epoch 27/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.3877 - accuracy: 0.8246 - val_loss: 0.5243 - val_accuracy: 0.7467
    Epoch 28/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.3791 - accuracy: 0.8318 - val_loss: 0.5112 - val_accuracy: 0.7512
    Epoch 29/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.3748 - accuracy: 0.8325 - val_loss: 0.5069 - val_accuracy: 0.7505
    Epoch 30/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.3662 - accuracy: 0.8362 - val_loss: 0.5117 - val_accuracy: 0.7545
    Epoch 31/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.3627 - accuracy: 0.8376 - val_loss: 0.5066 - val_accuracy: 0.7554
    Epoch 32/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.3581 - accuracy: 0.8400 - val_loss: 0.5086 - val_accuracy: 0.7543
    Epoch 33/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.3470 - accuracy: 0.8484 - val_loss: 0.5067 - val_accuracy: 0.7554
    Epoch 34/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.3447 - accuracy: 0.8440 - val_loss: 0.5138 - val_accuracy: 0.7554
    Epoch 35/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.3324 - accuracy: 0.8529 - val_loss: 0.5248 - val_accuracy: 0.7518
    Epoch 36/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.3349 - accuracy: 0.8522 - val_loss: 0.5257 - val_accuracy: 0.7528
    Epoch 37/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.3240 - accuracy: 0.8583 - val_loss: 0.5229 - val_accuracy: 0.7514
    Epoch 38/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.3315 - accuracy: 0.8557 - val_loss: 0.5084 - val_accuracy: 0.7533
    Epoch 39/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.3104 - accuracy: 0.8685 - val_loss: 0.5166 - val_accuracy: 0.7583
    Epoch 40/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.3058 - accuracy: 0.8681 - val_loss: 0.5260 - val_accuracy: 0.7590
    Epoch 41/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.3027 - accuracy: 0.8696 - val_loss: 0.5140 - val_accuracy: 0.7610
    Epoch 42/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.2995 - accuracy: 0.8716 - val_loss: 0.5150 - val_accuracy: 0.7613
    Epoch 43/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.2843 - accuracy: 0.8789 - val_loss: 0.5231 - val_accuracy: 0.7625
    Epoch 44/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.2902 - accuracy: 0.8755 - val_loss: 0.5136 - val_accuracy: 0.7598
    Epoch 45/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.2790 - accuracy: 0.8804 - val_loss: 0.5204 - val_accuracy: 0.7653
    Epoch 46/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.2729 - accuracy: 0.8869 - val_loss: 0.5251 - val_accuracy: 0.7571
    Epoch 47/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.2669 - accuracy: 0.8889 - val_loss: 0.5268 - val_accuracy: 0.7630
    Epoch 48/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.2640 - accuracy: 0.8911 - val_loss: 0.5307 - val_accuracy: 0.7623
    Epoch 49/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.2589 - accuracy: 0.8930 - val_loss: 0.5531 - val_accuracy: 0.7530
    Epoch 50/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.2625 - accuracy: 0.8889 - val_loss: 0.5372 - val_accuracy: 0.7625
    Epoch 51/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.2558 - accuracy: 0.8936 - val_loss: 0.5325 - val_accuracy: 0.7676
    Epoch 52/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.2415 - accuracy: 0.8972 - val_loss: 0.5413 - val_accuracy: 0.7646
    Epoch 53/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.2383 - accuracy: 0.9026 - val_loss: 0.5439 - val_accuracy: 0.7625
    Epoch 54/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.2348 - accuracy: 0.9024 - val_loss: 0.5393 - val_accuracy: 0.7638
    Epoch 55/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.2295 - accuracy: 0.9026 - val_loss: 0.5304 - val_accuracy: 0.7669
    Epoch 56/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.2258 - accuracy: 0.9077 - val_loss: 0.5547 - val_accuracy: 0.7589
    Epoch 57/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.2291 - accuracy: 0.9043 - val_loss: 0.5386 - val_accuracy: 0.7644
    Epoch 58/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.2168 - accuracy: 0.9132 - val_loss: 0.5554 - val_accuracy: 0.7661
    Epoch 59/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.2120 - accuracy: 0.9146 - val_loss: 0.5420 - val_accuracy: 0.7690
    Epoch 60/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.2055 - accuracy: 0.9172 - val_loss: 0.5614 - val_accuracy: 0.7634
    Epoch 61/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.2044 - accuracy: 0.9178 - val_loss: 0.5407 - val_accuracy: 0.7651
    Epoch 62/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.1977 - accuracy: 0.9217 - val_loss: 0.5625 - val_accuracy: 0.7691
    Epoch 63/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.1959 - accuracy: 0.9226 - val_loss: 0.5735 - val_accuracy: 0.7661
    Epoch 64/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.1919 - accuracy: 0.9233 - val_loss: 0.5544 - val_accuracy: 0.7676
    Epoch 65/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.1959 - accuracy: 0.9196 - val_loss: 0.5591 - val_accuracy: 0.7627
    Epoch 66/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.1785 - accuracy: 0.9324 - val_loss: 0.5681 - val_accuracy: 0.7699
    Epoch 67/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1805 - accuracy: 0.9280 - val_loss: 0.5827 - val_accuracy: 0.7627
    Epoch 68/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1763 - accuracy: 0.9289 - val_loss: 0.5730 - val_accuracy: 0.7684
    Epoch 69/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1687 - accuracy: 0.9367 - val_loss: 0.5774 - val_accuracy: 0.7672
    Epoch 70/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1674 - accuracy: 0.9342 - val_loss: 0.5726 - val_accuracy: 0.7697
    Epoch 71/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.1589 - accuracy: 0.9384 - val_loss: 0.5895 - val_accuracy: 0.7634
    Epoch 72/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.1648 - accuracy: 0.9356 - val_loss: 0.5876 - val_accuracy: 0.7699
    Epoch 73/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1594 - accuracy: 0.9382 - val_loss: 0.5930 - val_accuracy: 0.7712
    Epoch 74/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1553 - accuracy: 0.9380 - val_loss: 0.6131 - val_accuracy: 0.7594
    Epoch 75/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1597 - accuracy: 0.9347 - val_loss: 0.5800 - val_accuracy: 0.7691
    Epoch 76/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.1532 - accuracy: 0.9407 - val_loss: 0.5899 - val_accuracy: 0.7651
    Epoch 77/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.1469 - accuracy: 0.9444 - val_loss: 0.5941 - val_accuracy: 0.7693
    Epoch 78/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1528 - accuracy: 0.9407 - val_loss: 0.5985 - val_accuracy: 0.7686
    Epoch 79/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1417 - accuracy: 0.9459 - val_loss: 0.5940 - val_accuracy: 0.7720
    Epoch 80/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.1378 - accuracy: 0.9461 - val_loss: 0.6056 - val_accuracy: 0.7659
    Epoch 81/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.1411 - accuracy: 0.9459 - val_loss: 0.6063 - val_accuracy: 0.7716
    Epoch 82/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.1350 - accuracy: 0.9469 - val_loss: 0.6040 - val_accuracy: 0.7670
    Epoch 83/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1334 - accuracy: 0.9518 - val_loss: 0.5988 - val_accuracy: 0.7743
    Epoch 84/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1282 - accuracy: 0.9511 - val_loss: 0.6467 - val_accuracy: 0.7619
    Epoch 85/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1227 - accuracy: 0.9540 - val_loss: 0.6271 - val_accuracy: 0.7735
    Epoch 86/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1238 - accuracy: 0.9518 - val_loss: 0.6031 - val_accuracy: 0.7749
    Epoch 87/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1283 - accuracy: 0.9522 - val_loss: 0.6408 - val_accuracy: 0.7629
    Epoch 88/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1205 - accuracy: 0.9552 - val_loss: 0.6143 - val_accuracy: 0.7718
    Epoch 89/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1198 - accuracy: 0.9551 - val_loss: 0.6303 - val_accuracy: 0.7730
    Epoch 90/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1188 - accuracy: 0.9544 - val_loss: 0.6417 - val_accuracy: 0.7699
    Epoch 91/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1118 - accuracy: 0.9605 - val_loss: 0.6403 - val_accuracy: 0.7693
    Epoch 92/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1138 - accuracy: 0.9569 - val_loss: 0.6293 - val_accuracy: 0.7741
    Epoch 93/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.1106 - accuracy: 0.9590 - val_loss: 0.6900 - val_accuracy: 0.7659
    Epoch 94/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.1070 - accuracy: 0.9609 - val_loss: 0.6386 - val_accuracy: 0.7728
    Epoch 95/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.1075 - accuracy: 0.9607 - val_loss: 0.6572 - val_accuracy: 0.7735
    Epoch 96/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.1070 - accuracy: 0.9601 - val_loss: 0.6650 - val_accuracy: 0.7712
    Epoch 97/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.1060 - accuracy: 0.9600 - val_loss: 0.6528 - val_accuracy: 0.7728
    Epoch 98/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.0995 - accuracy: 0.9649 - val_loss: 0.6531 - val_accuracy: 0.7697
    Epoch 99/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0963 - accuracy: 0.9657 - val_loss: 0.6513 - val_accuracy: 0.7747
    Epoch 100/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0923 - accuracy: 0.9664 - val_loss: 0.6472 - val_accuracy: 0.7768
    Epoch 101/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0930 - accuracy: 0.9660 - val_loss: 0.6531 - val_accuracy: 0.7762
    Epoch 102/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0938 - accuracy: 0.9651 - val_loss: 0.6497 - val_accuracy: 0.7781
    Epoch 103/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0915 - accuracy: 0.9681 - val_loss: 0.6610 - val_accuracy: 0.7752
    Epoch 104/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.0912 - accuracy: 0.9682 - val_loss: 0.6551 - val_accuracy: 0.7766
    Epoch 105/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.0880 - accuracy: 0.9685 - val_loss: 0.6686 - val_accuracy: 0.7758
    Epoch 106/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.0847 - accuracy: 0.9705 - val_loss: 0.6681 - val_accuracy: 0.7724
    Epoch 107/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.0855 - accuracy: 0.9690 - val_loss: 0.6877 - val_accuracy: 0.7726
    Epoch 108/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.0883 - accuracy: 0.9669 - val_loss: 0.6670 - val_accuracy: 0.7764
    Epoch 109/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.0858 - accuracy: 0.9700 - val_loss: 0.6929 - val_accuracy: 0.7714
    Epoch 110/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.0821 - accuracy: 0.9704 - val_loss: 0.7237 - val_accuracy: 0.7650
    Epoch 111/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.0770 - accuracy: 0.9732 - val_loss: 0.6836 - val_accuracy: 0.7710
    Epoch 112/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.0769 - accuracy: 0.9713 - val_loss: 0.7192 - val_accuracy: 0.7709
    Epoch 113/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.0750 - accuracy: 0.9736 - val_loss: 0.7226 - val_accuracy: 0.7710
    Epoch 114/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.0764 - accuracy: 0.9711 - val_loss: 0.7069 - val_accuracy: 0.7731
    Epoch 115/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.0751 - accuracy: 0.9736 - val_loss: 0.7194 - val_accuracy: 0.7682
    Epoch 116/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.0730 - accuracy: 0.9753 - val_loss: 0.6966 - val_accuracy: 0.7741
    Epoch 117/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.0720 - accuracy: 0.9744 - val_loss: 0.7272 - val_accuracy: 0.7714
    Epoch 118/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.0710 - accuracy: 0.9742 - val_loss: 0.7206 - val_accuracy: 0.7718
    Epoch 119/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.0705 - accuracy: 0.9748 - val_loss: 0.7289 - val_accuracy: 0.7684
    Epoch 120/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.0718 - accuracy: 0.9741 - val_loss: 0.7023 - val_accuracy: 0.7781
    Epoch 121/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.0702 - accuracy: 0.9758 - val_loss: 0.7309 - val_accuracy: 0.7724
    Epoch 122/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.0684 - accuracy: 0.9763 - val_loss: 0.7314 - val_accuracy: 0.7737
    Epoch 123/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.0684 - accuracy: 0.9760 - val_loss: 0.7207 - val_accuracy: 0.7657
    Epoch 124/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.0705 - accuracy: 0.9740 - val_loss: 0.7334 - val_accuracy: 0.7699
    Epoch 125/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.0648 - accuracy: 0.9772 - val_loss: 0.7289 - val_accuracy: 0.7731
    Epoch 126/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.0619 - accuracy: 0.9781 - val_loss: 0.7250 - val_accuracy: 0.7728
    Epoch 127/200
    123/123 [==============================] - 5s 37ms/step - loss: 0.0624 - accuracy: 0.9790 - val_loss: 0.7589 - val_accuracy: 0.7678
    Epoch 128/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.0669 - accuracy: 0.9754 - val_loss: 0.7276 - val_accuracy: 0.7716
    Epoch 129/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.0600 - accuracy: 0.9779 - val_loss: 0.7151 - val_accuracy: 0.7707
    Epoch 130/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.0620 - accuracy: 0.9789 - val_loss: 0.7452 - val_accuracy: 0.7747
    Epoch 131/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.0603 - accuracy: 0.9795 - val_loss: 0.7126 - val_accuracy: 0.7722
    Epoch 132/200
    123/123 [==============================] - 5s 39ms/step - loss: 0.0593 - accuracy: 0.9807 - val_loss: 0.7216 - val_accuracy: 0.7718
    Epoch 133/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.0558 - accuracy: 0.9807 - val_loss: 0.7366 - val_accuracy: 0.7762
    Epoch 134/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.0619 - accuracy: 0.9785 - val_loss: 0.7400 - val_accuracy: 0.7739
    Epoch 135/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.0508 - accuracy: 0.9833 - val_loss: 0.7682 - val_accuracy: 0.7634
    Epoch 136/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.0539 - accuracy: 0.9826 - val_loss: 0.7446 - val_accuracy: 0.7733
    Epoch 137/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.0553 - accuracy: 0.9808 - val_loss: 0.7418 - val_accuracy: 0.7773
    Epoch 138/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.0526 - accuracy: 0.9822 - val_loss: 0.7460 - val_accuracy: 0.7758
    Epoch 139/200
    123/123 [==============================] - 5s 38ms/step - loss: 0.0513 - accuracy: 0.9829 - val_loss: 0.7558 - val_accuracy: 0.7781
    Epoch 140/200
    123/123 [==============================] - 4s 37ms/step - loss: 0.0496 - accuracy: 0.9833 - val_loss: 0.7945 - val_accuracy: 0.7718
    Epoch 141/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0526 - accuracy: 0.9818 - val_loss: 0.7279 - val_accuracy: 0.7771
    Epoch 142/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0485 - accuracy: 0.9842 - val_loss: 0.7653 - val_accuracy: 0.7752
    Epoch 143/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0556 - accuracy: 0.9810 - val_loss: 0.7348 - val_accuracy: 0.7720
    Epoch 144/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0556 - accuracy: 0.9806 - val_loss: 0.7421 - val_accuracy: 0.7712
    Epoch 145/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0511 - accuracy: 0.9809 - val_loss: 0.7757 - val_accuracy: 0.7741
    Epoch 146/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0481 - accuracy: 0.9832 - val_loss: 0.7564 - val_accuracy: 0.7712
    Epoch 147/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0500 - accuracy: 0.9824 - val_loss: 0.7648 - val_accuracy: 0.7697
    Epoch 148/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0462 - accuracy: 0.9834 - val_loss: 0.7673 - val_accuracy: 0.7720
    Epoch 149/200
    123/123 [==============================] - 4s 36ms/step - loss: 0.0479 - accuracy: 0.9843 - val_loss: 0.7442 - val_accuracy: 0.7752
    Epoch 150/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0452 - accuracy: 0.9860 - val_loss: 0.8148 - val_accuracy: 0.7703
    Epoch 151/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0477 - accuracy: 0.9829 - val_loss: 0.7818 - val_accuracy: 0.7766
    Epoch 152/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0445 - accuracy: 0.9841 - val_loss: 0.7333 - val_accuracy: 0.7750
    Epoch 153/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0359 - accuracy: 0.9902 - val_loss: 0.8055 - val_accuracy: 0.7766
    Epoch 154/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0485 - accuracy: 0.9833 - val_loss: 0.7644 - val_accuracy: 0.7724
    Epoch 155/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0508 - accuracy: 0.9828 - val_loss: 0.7576 - val_accuracy: 0.7762
    Epoch 156/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0458 - accuracy: 0.9855 - val_loss: 0.7868 - val_accuracy: 0.7794
    Epoch 157/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0426 - accuracy: 0.9850 - val_loss: 0.8222 - val_accuracy: 0.7726
    Epoch 158/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0443 - accuracy: 0.9854 - val_loss: 0.7700 - val_accuracy: 0.7707
    Epoch 159/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0437 - accuracy: 0.9857 - val_loss: 0.7561 - val_accuracy: 0.7750
    Epoch 160/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0417 - accuracy: 0.9864 - val_loss: 0.7912 - val_accuracy: 0.7737
    Epoch 161/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0387 - accuracy: 0.9873 - val_loss: 0.7994 - val_accuracy: 0.7731
    Epoch 162/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0392 - accuracy: 0.9876 - val_loss: 0.8054 - val_accuracy: 0.7739
    Epoch 163/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0379 - accuracy: 0.9878 - val_loss: 0.7870 - val_accuracy: 0.7750
    Epoch 164/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0353 - accuracy: 0.9896 - val_loss: 0.8040 - val_accuracy: 0.7724
    Epoch 165/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0358 - accuracy: 0.9891 - val_loss: 0.8352 - val_accuracy: 0.7773
    Epoch 166/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0374 - accuracy: 0.9869 - val_loss: 0.8137 - val_accuracy: 0.7756
    Epoch 167/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0355 - accuracy: 0.9888 - val_loss: 0.7922 - val_accuracy: 0.7768
    Epoch 168/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0356 - accuracy: 0.9880 - val_loss: 0.8054 - val_accuracy: 0.7749
    Epoch 169/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0417 - accuracy: 0.9856 - val_loss: 0.7684 - val_accuracy: 0.7758
    Epoch 170/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0351 - accuracy: 0.9886 - val_loss: 0.7786 - val_accuracy: 0.7783
    Epoch 171/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0383 - accuracy: 0.9876 - val_loss: 0.7697 - val_accuracy: 0.7739
    Epoch 172/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0361 - accuracy: 0.9887 - val_loss: 0.7997 - val_accuracy: 0.7775
    Epoch 173/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0352 - accuracy: 0.9888 - val_loss: 0.8144 - val_accuracy: 0.7783
    Epoch 174/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0346 - accuracy: 0.9890 - val_loss: 0.8418 - val_accuracy: 0.7750
    Epoch 175/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0321 - accuracy: 0.9895 - val_loss: 0.8243 - val_accuracy: 0.7730
    Epoch 176/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0317 - accuracy: 0.9905 - val_loss: 0.7620 - val_accuracy: 0.7762
    Epoch 177/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0340 - accuracy: 0.9890 - val_loss: 0.8056 - val_accuracy: 0.7726
    Epoch 178/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0325 - accuracy: 0.9896 - val_loss: 0.8210 - val_accuracy: 0.7749
    Epoch 179/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0324 - accuracy: 0.9900 - val_loss: 0.8070 - val_accuracy: 0.7766
    Epoch 180/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0319 - accuracy: 0.9904 - val_loss: 0.7985 - val_accuracy: 0.7743
    Epoch 181/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0295 - accuracy: 0.9900 - val_loss: 0.8606 - val_accuracy: 0.7754
    Epoch 182/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0336 - accuracy: 0.9892 - val_loss: 0.8056 - val_accuracy: 0.7770
    Epoch 183/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0307 - accuracy: 0.9902 - val_loss: 0.8177 - val_accuracy: 0.7739
    Epoch 184/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0316 - accuracy: 0.9894 - val_loss: 0.8365 - val_accuracy: 0.7735
    Epoch 185/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0309 - accuracy: 0.9897 - val_loss: 0.7995 - val_accuracy: 0.7745
    Epoch 186/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0292 - accuracy: 0.9908 - val_loss: 0.8736 - val_accuracy: 0.7750
    Epoch 187/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0319 - accuracy: 0.9897 - val_loss: 0.8168 - val_accuracy: 0.7739
    Epoch 188/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0248 - accuracy: 0.9929 - val_loss: 0.8341 - val_accuracy: 0.7808
    Epoch 189/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0264 - accuracy: 0.9918 - val_loss: 0.8642 - val_accuracy: 0.7701
    Epoch 190/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0292 - accuracy: 0.9904 - val_loss: 0.8021 - val_accuracy: 0.7777
    Epoch 191/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0258 - accuracy: 0.9918 - val_loss: 0.8644 - val_accuracy: 0.7789
    Epoch 192/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0276 - accuracy: 0.9904 - val_loss: 0.8415 - val_accuracy: 0.7783
    Epoch 193/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0277 - accuracy: 0.9913 - val_loss: 0.8053 - val_accuracy: 0.7771
    Epoch 194/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0268 - accuracy: 0.9918 - val_loss: 0.8498 - val_accuracy: 0.7739
    Epoch 195/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0272 - accuracy: 0.9910 - val_loss: 0.7959 - val_accuracy: 0.7762
    Epoch 196/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0275 - accuracy: 0.9909 - val_loss: 0.8180 - val_accuracy: 0.7768
    Epoch 197/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0272 - accuracy: 0.9901 - val_loss: 0.8445 - val_accuracy: 0.7766
    Epoch 198/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0224 - accuracy: 0.9943 - val_loss: 0.8614 - val_accuracy: 0.7779
    Epoch 199/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0270 - accuracy: 0.9917 - val_loss: 0.8285 - val_accuracy: 0.7710
    Epoch 200/200
    123/123 [==============================] - 4s 35ms/step - loss: 0.0255 - accuracy: 0.9918 - val_loss: 0.8527 - val_accuracy: 0.7754
    Time elapsed : 0:14:43.835623
    


```python
result = model.evaluate(norm_test_x_data.reshape(-1, 80, 80, 1),
                        test_t_data.reshape(-1, 1))
print(result)
#        loss           accuracy
```

    235/235 [==============================] - 1s 5ms/step - loss: 0.8762 - accuracy: 0.7701
    [0.8761647939682007, 0.7701333165168762]
    


```python
model.save('D:/jupyter_home/data/kaggle/cat_dog/full_data_model/full_data_model.h5')
```


```python
# history 객체를 이용해 학습과정에 대한 그래프 그리기
train_acc = history.history['accuracy']
train_loss = history.history['loss']

valid_acc = history.history['val_accuracy']
valid_loss = history.history['val_loss']

figure = plt.figure()
ax1 = figure.add_subplot(1, 2, 1)
ax2 = figure.add_subplot(1, 2, 2)

ax1.plot(train_acc, color='b', label='training accuracy')
ax1.plot(valid_acc, color='r', label='valid accuracy')
ax1.legend()

ax2.plot(train_loss, color='b', label='training loss')
ax2.plot(valid_loss, color='r', label='valid loss')
ax2.legend()

plt.tight_layout()
plt.show()
```


    
![png](/Machine-Learning/images/0418/output_11_0.png)
    

