# Tensorflow 버전 1.15 vs. 2.x


```python
import tensorflow as tf

W = tf.random.normal([1], dtype=tf.float32)  # tensorflow node

# 1.15 버전에서 W의 값을 알아내려면 session을 통해서 node를 실행시켜 값을 얻어야 함
# 2.x 버전은 eager execution(즉시실행모드) 지원 - session 필요 X, 일반적인 프로그래밍 하는 것처럼 사용 가능
print(W.numpy())   # [-0.7210334]

# sess.run(tf.gloabal_variables_initializer())  # 초기화 코드 불필요
# placeholder X
```

    [-0.64326286]
    

## Keras의 model 만들기
import tensorflow as tf

model = tf.keras.models.Sequential()

- model.add()로 layer 추가
    - model.add(input layer)
    - model.add(output layter)

- loss의 종류와 optimizer 종류 설정
    - model.compile()

- 학습 (sklearn과 유사한 형태로 사용)
    - model.fit()

- 평가와 predict
    - model.evaluate()  => 모델평가
    - model.predict()   => 예측값 도출

- 모델 저장
    - model.save()

# Tensorflow 2.x 버전으로  MNIST 예제 구현하기


```python
import tensorflow as tf

print(tf.__version__)  
```

    2.3.0
    


```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential      # keras model
from tensorflow.keras.layers import Flatten, Dense  # Flatten(Input Layer)
                                                     # Dense(Output Layer)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Raw Data Loading

df = pd.read_csv('./data/kaggle/mnist/train.csv')
display(df.head())
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>



```python
# Data Split
train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(df.drop('label', axis=1, inplace=False),
                 df['label'],
                 test_size=0.3,
                 random_state=1,
                 stratify=df['label'])
# 정규화
scaler = MinMaxScaler()
scaler.fit(train_x_data)

norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)
```


```python
# model 생성
model = Sequential()  # 2가지 모델 중 하나

# layer 추가 - input layer
model.add(Flatten(input_shape=(norm_train_x_data.shape[1],)))  # 튜플로 독립 변수의 개수를 2차원으로 표현

# layer 추가 - output layer
model.add(Dense(units=10,  # FC(Fully Connected) layer 구현 - units : logistic의 개수
                activation='softmax'))  # activation 함수 (multinomial이므로 sigmoid가 아닌 softmax 사용)

print(model.summary())  # total params : 구해야 할 W의 개수
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    dense (Dense)                (None, 10)                7850      
    =================================================================
    Total params: 7,850
    Trainable params: 7,850
    Non-trainable params: 0
    _________________________________________________________________
    None
    

### loss
- linear regression : linear
- binary classification : binary_crossentropy
- multinomial classification : categorical_crossentropy(onehot encoding 처리 해야 함)
- multinomial classification : sparse_categorical_crossentropy(onehot encoding 처리 필요 없음)


```python
# model compile
# 사용할 loss 함수, optimizer(알고리즘) 지정
from tensorflow.keras.optimizers import SGD


model.compile(optimizer=SGD(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 학습결과를 변수에 저장
history = model.fit(norm_train_x_data,
                    train_t_data,  # categorical_crossentropy일 경우 onehot encoding 처리 한 train_t_data
                    epochs=100,
                    batch_size=100,
                    verbose=1,  # 진행 결과 출력
                    validation_split=0.2)  # 들어온 데이터를 (train용, validation용)8:2로 나눠서 씀

# accuracy와 val_accuracy 간 차이가 크면 overfitting이 발생한 것
```

    Epoch 1/100
    236/236 [==============================] - 0s 1ms/step - loss: 2.3116 - accuracy: 0.1331 - val_loss: 2.1821 - val_accuracy: 0.2060
    Epoch 2/100
    236/236 [==============================] - 0s 590us/step - loss: 2.0736 - accuracy: 0.2838 - val_loss: 1.9712 - val_accuracy: 0.3801
    Epoch 3/100
    236/236 [==============================] - 0s 585us/step - loss: 1.8806 - accuracy: 0.4629 - val_loss: 1.7956 - val_accuracy: 0.5371
    Epoch 4/100
    236/236 [==============================] - 0s 590us/step - loss: 1.7195 - accuracy: 0.5842 - val_loss: 1.6484 - val_accuracy: 0.6221
    Epoch 5/100
    236/236 [==============================] - 0s 611us/step - loss: 1.5846 - accuracy: 0.6493 - val_loss: 1.5248 - val_accuracy: 0.6723
    Epoch 6/100
    236/236 [==============================] - 0s 602us/step - loss: 1.4713 - accuracy: 0.6897 - val_loss: 1.4208 - val_accuracy: 0.7024
    Epoch 7/100
    236/236 [==============================] - 0s 590us/step - loss: 1.3759 - accuracy: 0.7158 - val_loss: 1.3328 - val_accuracy: 0.7252
    Epoch 8/100
    236/236 [==============================] - 0s 611us/step - loss: 1.2948 - accuracy: 0.7357 - val_loss: 1.2578 - val_accuracy: 0.7435
    Epoch 9/100
    236/236 [==============================] - 0s 640us/step - loss: 1.2255 - accuracy: 0.7485 - val_loss: 1.1934 - val_accuracy: 0.7588
    Epoch 10/100
    236/236 [==============================] - 0s 636us/step - loss: 1.1657 - accuracy: 0.7597 - val_loss: 1.1377 - val_accuracy: 0.7689
    Epoch 11/100
    236/236 [==============================] - 0s 649us/step - loss: 1.1138 - accuracy: 0.7708 - val_loss: 1.0890 - val_accuracy: 0.7779
    Epoch 12/100
    236/236 [==============================] - 0s 602us/step - loss: 1.0683 - accuracy: 0.7794 - val_loss: 1.0463 - val_accuracy: 0.7854
    Epoch 13/100
    236/236 [==============================] - 0s 590us/step - loss: 1.0281 - accuracy: 0.7863 - val_loss: 1.0084 - val_accuracy: 0.7912
    Epoch 14/100
    236/236 [==============================] - 0s 590us/step - loss: 0.9924 - accuracy: 0.7927 - val_loss: 0.9745 - val_accuracy: 0.7976
    Epoch 15/100
    236/236 [==============================] - 0s 606us/step - loss: 0.9604 - accuracy: 0.7980 - val_loss: 0.9441 - val_accuracy: 0.8039
    Epoch 16/100
    236/236 [==============================] - 0s 602us/step - loss: 0.9316 - accuracy: 0.8028 - val_loss: 0.9167 - val_accuracy: 0.8083
    Epoch 17/100
    236/236 [==============================] - 0s 602us/step - loss: 0.9055 - accuracy: 0.8071 - val_loss: 0.8919 - val_accuracy: 0.8136
    Epoch 18/100
    236/236 [==============================] - 0s 590us/step - loss: 0.8818 - accuracy: 0.8111 - val_loss: 0.8691 - val_accuracy: 0.8165
    Epoch 19/100
    236/236 [==============================] - 0s 628us/step - loss: 0.8601 - accuracy: 0.8145 - val_loss: 0.8483 - val_accuracy: 0.8214
    Epoch 20/100
    236/236 [==============================] - 0s 606us/step - loss: 0.8402 - accuracy: 0.8176 - val_loss: 0.8294 - val_accuracy: 0.8228
    Epoch 21/100
    236/236 [==============================] - 0s 585us/step - loss: 0.8219 - accuracy: 0.8212 - val_loss: 0.8118 - val_accuracy: 0.8260
    Epoch 22/100
    236/236 [==============================] - 0s 590us/step - loss: 0.8050 - accuracy: 0.8241 - val_loss: 0.7955 - val_accuracy: 0.8282
    Epoch 23/100
    236/236 [==============================] - 0s 598us/step - loss: 0.7892 - accuracy: 0.8271 - val_loss: 0.7804 - val_accuracy: 0.8304
    Epoch 24/100
    236/236 [==============================] - 0s 585us/step - loss: 0.7746 - accuracy: 0.8287 - val_loss: 0.7663 - val_accuracy: 0.8323
    Epoch 25/100
    236/236 [==============================] - 0s 590us/step - loss: 0.7609 - accuracy: 0.8308 - val_loss: 0.7532 - val_accuracy: 0.8340
    Epoch 26/100
    236/236 [==============================] - 0s 640us/step - loss: 0.7482 - accuracy: 0.8332 - val_loss: 0.7410 - val_accuracy: 0.8364
    Epoch 27/100
    236/236 [==============================] - 0s 636us/step - loss: 0.7362 - accuracy: 0.8347 - val_loss: 0.7294 - val_accuracy: 0.8384
    Epoch 28/100
    236/236 [==============================] - 0s 628us/step - loss: 0.7249 - accuracy: 0.8363 - val_loss: 0.7185 - val_accuracy: 0.8401
    Epoch 29/100
    236/236 [==============================] - 0s 615us/step - loss: 0.7143 - accuracy: 0.8379 - val_loss: 0.7083 - val_accuracy: 0.8415
    Epoch 30/100
    236/236 [==============================] - 0s 602us/step - loss: 0.7043 - accuracy: 0.8392 - val_loss: 0.6986 - val_accuracy: 0.8432
    Epoch 31/100
    236/236 [==============================] - 0s 594us/step - loss: 0.6948 - accuracy: 0.8410 - val_loss: 0.6895 - val_accuracy: 0.8449
    Epoch 32/100
    236/236 [==============================] - 0s 640us/step - loss: 0.6859 - accuracy: 0.8423 - val_loss: 0.6808 - val_accuracy: 0.8457
    Epoch 33/100
    236/236 [==============================] - 0s 645us/step - loss: 0.6774 - accuracy: 0.8440 - val_loss: 0.6726 - val_accuracy: 0.8469
    Epoch 34/100
    236/236 [==============================] - 0s 615us/step - loss: 0.6693 - accuracy: 0.8452 - val_loss: 0.6649 - val_accuracy: 0.8480
    Epoch 35/100
    236/236 [==============================] - 0s 606us/step - loss: 0.6616 - accuracy: 0.8465 - val_loss: 0.6574 - val_accuracy: 0.8483
    Epoch 36/100
    236/236 [==============================] - 0s 645us/step - loss: 0.6543 - accuracy: 0.8474 - val_loss: 0.6503 - val_accuracy: 0.8502
    Epoch 37/100
    236/236 [==============================] - 0s 628us/step - loss: 0.6473 - accuracy: 0.8484 - val_loss: 0.6436 - val_accuracy: 0.8509
    Epoch 38/100
    236/236 [==============================] - 0s 649us/step - loss: 0.6406 - accuracy: 0.8496 - val_loss: 0.6371 - val_accuracy: 0.8527
    Epoch 39/100
    236/236 [==============================] - 0s 636us/step - loss: 0.6342 - accuracy: 0.8507 - val_loss: 0.6310 - val_accuracy: 0.8537
    Epoch 40/100
    236/236 [==============================] - 0s 623us/step - loss: 0.6280 - accuracy: 0.8517 - val_loss: 0.6251 - val_accuracy: 0.8549
    Epoch 41/100
    236/236 [==============================] - 0s 606us/step - loss: 0.6222 - accuracy: 0.8529 - val_loss: 0.6194 - val_accuracy: 0.8556
    Epoch 42/100
    236/236 [==============================] - 0s 632us/step - loss: 0.6165 - accuracy: 0.8540 - val_loss: 0.6139 - val_accuracy: 0.8565
    Epoch 43/100
    236/236 [==============================] - 0s 640us/step - loss: 0.6111 - accuracy: 0.8551 - val_loss: 0.6087 - val_accuracy: 0.8568
    Epoch 44/100
    236/236 [==============================] - 0s 649us/step - loss: 0.6059 - accuracy: 0.8561 - val_loss: 0.6037 - val_accuracy: 0.8568
    Epoch 45/100
    236/236 [==============================] - 0s 598us/step - loss: 0.6009 - accuracy: 0.8571 - val_loss: 0.5988 - val_accuracy: 0.8590
    Epoch 46/100
    236/236 [==============================] - 0s 602us/step - loss: 0.5960 - accuracy: 0.8580 - val_loss: 0.5941 - val_accuracy: 0.8590
    Epoch 47/100
    236/236 [==============================] - 0s 636us/step - loss: 0.5914 - accuracy: 0.8587 - val_loss: 0.5897 - val_accuracy: 0.8602
    Epoch 48/100
    236/236 [==============================] - 0s 636us/step - loss: 0.5869 - accuracy: 0.8594 - val_loss: 0.5853 - val_accuracy: 0.8612
    Epoch 49/100
    236/236 [==============================] - 0s 649us/step - loss: 0.5825 - accuracy: 0.8600 - val_loss: 0.5812 - val_accuracy: 0.8624
    Epoch 50/100
    236/236 [==============================] - 0s 632us/step - loss: 0.5783 - accuracy: 0.8604 - val_loss: 0.5771 - val_accuracy: 0.8631
    Epoch 51/100
    236/236 [==============================] - 0s 611us/step - loss: 0.5743 - accuracy: 0.8615 - val_loss: 0.5732 - val_accuracy: 0.8636
    Epoch 52/100
    236/236 [==============================] - 0s 619us/step - loss: 0.5704 - accuracy: 0.8625 - val_loss: 0.5694 - val_accuracy: 0.8643
    Epoch 53/100
    236/236 [==============================] - 0s 611us/step - loss: 0.5666 - accuracy: 0.8630 - val_loss: 0.5657 - val_accuracy: 0.8645
    Epoch 54/100
    236/236 [==============================] - 0s 636us/step - loss: 0.5629 - accuracy: 0.8636 - val_loss: 0.5622 - val_accuracy: 0.8651
    Epoch 55/100
    236/236 [==============================] - 0s 615us/step - loss: 0.5593 - accuracy: 0.8645 - val_loss: 0.5588 - val_accuracy: 0.8658
    Epoch 56/100
    236/236 [==============================] - 0s 611us/step - loss: 0.5558 - accuracy: 0.8650 - val_loss: 0.5554 - val_accuracy: 0.8667
    Epoch 57/100
    236/236 [==============================] - 0s 598us/step - loss: 0.5525 - accuracy: 0.8658 - val_loss: 0.5522 - val_accuracy: 0.8670
    Epoch 58/100
    236/236 [==============================] - 0s 606us/step - loss: 0.5492 - accuracy: 0.8665 - val_loss: 0.5491 - val_accuracy: 0.8672
    Epoch 59/100
    236/236 [==============================] - 0s 666us/step - loss: 0.5461 - accuracy: 0.8668 - val_loss: 0.5461 - val_accuracy: 0.8673
    Epoch 60/100
    236/236 [==============================] - 0s 738us/step - loss: 0.5430 - accuracy: 0.8676 - val_loss: 0.5431 - val_accuracy: 0.8673
    Epoch 61/100
    236/236 [==============================] - 0s 640us/step - loss: 0.5400 - accuracy: 0.8678 - val_loss: 0.5402 - val_accuracy: 0.8682
    Epoch 62/100
    236/236 [==============================] - 0s 615us/step - loss: 0.5371 - accuracy: 0.8686 - val_loss: 0.5374 - val_accuracy: 0.8684
    Epoch 63/100
    236/236 [==============================] - 0s 611us/step - loss: 0.5342 - accuracy: 0.8691 - val_loss: 0.5347 - val_accuracy: 0.8685
    Epoch 64/100
    236/236 [==============================] - 0s 606us/step - loss: 0.5315 - accuracy: 0.8696 - val_loss: 0.5320 - val_accuracy: 0.8685
    Epoch 65/100
    236/236 [==============================] - 0s 619us/step - loss: 0.5288 - accuracy: 0.8703 - val_loss: 0.5294 - val_accuracy: 0.8689
    Epoch 66/100
    236/236 [==============================] - 0s 700us/step - loss: 0.5261 - accuracy: 0.8708 - val_loss: 0.5269 - val_accuracy: 0.8697
    Epoch 67/100
    236/236 [==============================] - 0s 776us/step - loss: 0.5236 - accuracy: 0.8713 - val_loss: 0.5244 - val_accuracy: 0.8702
    Epoch 68/100
    236/236 [==============================] - 0s 886us/step - loss: 0.5211 - accuracy: 0.8719 - val_loss: 0.5220 - val_accuracy: 0.8704
    Epoch 69/100
    236/236 [==============================] - 0s 780us/step - loss: 0.5186 - accuracy: 0.8722 - val_loss: 0.5197 - val_accuracy: 0.8704
    Epoch 70/100
    236/236 [==============================] - 0s 763us/step - loss: 0.5163 - accuracy: 0.8728 - val_loss: 0.5174 - val_accuracy: 0.8707
    Epoch 71/100
    236/236 [==============================] - 0s 632us/step - loss: 0.5139 - accuracy: 0.8729 - val_loss: 0.5152 - val_accuracy: 0.8706
    Epoch 72/100
    236/236 [==============================] - 0s 670us/step - loss: 0.5117 - accuracy: 0.8734 - val_loss: 0.5130 - val_accuracy: 0.8709
    Epoch 73/100
    236/236 [==============================] - 0s 717us/step - loss: 0.5094 - accuracy: 0.8739 - val_loss: 0.5109 - val_accuracy: 0.8716
    Epoch 74/100
    236/236 [==============================] - 0s 674us/step - loss: 0.5073 - accuracy: 0.8741 - val_loss: 0.5088 - val_accuracy: 0.8716
    Epoch 75/100
    236/236 [==============================] - 0s 713us/step - loss: 0.5051 - accuracy: 0.8743 - val_loss: 0.5068 - val_accuracy: 0.8721
    Epoch 76/100
    236/236 [==============================] - 0s 700us/step - loss: 0.5031 - accuracy: 0.8744 - val_loss: 0.5048 - val_accuracy: 0.8724
    Epoch 77/100
    236/236 [==============================] - 0s 657us/step - loss: 0.5010 - accuracy: 0.8749 - val_loss: 0.5029 - val_accuracy: 0.8728
    Epoch 78/100
    236/236 [==============================] - 0s 666us/step - loss: 0.4990 - accuracy: 0.8751 - val_loss: 0.5010 - val_accuracy: 0.8736
    Epoch 79/100
    236/236 [==============================] - 0s 657us/step - loss: 0.4971 - accuracy: 0.8756 - val_loss: 0.4991 - val_accuracy: 0.8736
    Epoch 80/100
    236/236 [==============================] - 0s 636us/step - loss: 0.4952 - accuracy: 0.8760 - val_loss: 0.4973 - val_accuracy: 0.8735
    Epoch 81/100
    236/236 [==============================] - 0s 649us/step - loss: 0.4933 - accuracy: 0.8764 - val_loss: 0.4955 - val_accuracy: 0.8736
    Epoch 82/100
    236/236 [==============================] - 0s 640us/step - loss: 0.4915 - accuracy: 0.8768 - val_loss: 0.4937 - val_accuracy: 0.8740
    Epoch 83/100
    236/236 [==============================] - 0s 619us/step - loss: 0.4897 - accuracy: 0.8770 - val_loss: 0.4920 - val_accuracy: 0.8741
    Epoch 84/100
    236/236 [==============================] - 0s 606us/step - loss: 0.4880 - accuracy: 0.8773 - val_loss: 0.4903 - val_accuracy: 0.8748
    Epoch 85/100
    236/236 [==============================] - 0s 623us/step - loss: 0.4862 - accuracy: 0.8776 - val_loss: 0.4887 - val_accuracy: 0.8745
    Epoch 86/100
    236/236 [==============================] - 0s 636us/step - loss: 0.4845 - accuracy: 0.8777 - val_loss: 0.4871 - val_accuracy: 0.8750
    Epoch 87/100
    236/236 [==============================] - 0s 848us/step - loss: 0.4829 - accuracy: 0.8779 - val_loss: 0.4855 - val_accuracy: 0.8752
    Epoch 88/100
    236/236 [==============================] - 0s 729us/step - loss: 0.4812 - accuracy: 0.8782 - val_loss: 0.4839 - val_accuracy: 0.8750
    Epoch 89/100
    236/236 [==============================] - 0s 725us/step - loss: 0.4797 - accuracy: 0.8784 - val_loss: 0.4824 - val_accuracy: 0.8755
    Epoch 90/100
    236/236 [==============================] - 0s 687us/step - loss: 0.4781 - accuracy: 0.8785 - val_loss: 0.4809 - val_accuracy: 0.8757
    Epoch 91/100
    236/236 [==============================] - 0s 636us/step - loss: 0.4765 - accuracy: 0.8787 - val_loss: 0.4794 - val_accuracy: 0.8760
    Epoch 92/100
    236/236 [==============================] - 0s 636us/step - loss: 0.4750 - accuracy: 0.8790 - val_loss: 0.4780 - val_accuracy: 0.8769
    Epoch 93/100
    236/236 [==============================] - 0s 645us/step - loss: 0.4735 - accuracy: 0.8793 - val_loss: 0.4766 - val_accuracy: 0.8774
    Epoch 94/100
    236/236 [==============================] - 0s 645us/step - loss: 0.4721 - accuracy: 0.8795 - val_loss: 0.4752 - val_accuracy: 0.8769
    Epoch 95/100
    236/236 [==============================] - 0s 666us/step - loss: 0.4706 - accuracy: 0.8798 - val_loss: 0.4738 - val_accuracy: 0.8769
    Epoch 96/100
    236/236 [==============================] - 0s 683us/step - loss: 0.4692 - accuracy: 0.8802 - val_loss: 0.4725 - val_accuracy: 0.8770
    Epoch 97/100
    236/236 [==============================] - 0s 666us/step - loss: 0.4678 - accuracy: 0.8803 - val_loss: 0.4712 - val_accuracy: 0.8777
    Epoch 98/100
    236/236 [==============================] - 0s 645us/step - loss: 0.4665 - accuracy: 0.8806 - val_loss: 0.4699 - val_accuracy: 0.8782
    Epoch 99/100
    236/236 [==============================] - 0s 653us/step - loss: 0.4651 - accuracy: 0.8808 - val_loss: 0.4686 - val_accuracy: 0.8787
    Epoch 100/100
    236/236 [==============================] - 0s 602us/step - loss: 0.4638 - accuracy: 0.8811 - val_loss: 0.4674 - val_accuracy: 0.8786
    


```python
print(model.evaluate(norm_test_x_data, test_t_data))
#        loss               accuracy
```

    394/394 [==============================] - 0s 335us/step - loss: 0.4840 - accuracy: 0.8748
    [0.48397016525268555, 0.8748412728309631]
    

### 학습한 후 모델 저장하기

1. 모델을 저장할 때 모델 구조와 계산된 W,b를 같이 저장
- 장점 => 편함
- 단점 => 크기가 큼

2. 모델을 저장할 때 모델 구조는 저장하지 않고 W,b만 저장 (일반적으로 많이 쓰는 방식)
- 장점 => 크기가 작음 
- 단점 => 사용하려면 모델을 먼저 만들고 W,b를 로딩해야 함


```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential      # keras model
from tensorflow.keras.layers import Flatten, Dense  # Flatten(Input Layer)
                                                     # Dense(Output Layer)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint

# Raw Data Loading
df = pd.read_csv('./data/kaggle/mnist/train.csv')

# Data Split
# 기존에는 test_x_data, test_t_data를 validation 용도로 사용
# => 이제 test 용도로 사용
# 최종 모델 성능평가를 위해서 1번만 사용

# validation
# keras는 학습할 때 train data를 일정부분 나누어서 자체 validation이 가능
# keras 기능을 이용해서 validation 처리

train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(df.drop('label', axis=1, inplace=False),
                 df['label'],
                 test_size=0.3,
                 random_state=1,
                 stratify=df['label'])

# 정규화
scaler = MinMaxScaler()
scaler.fit(train_x_data)

norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)

# sparse_categorical_crossentropy로 loss함수 지정 
# => label에 대한 one-hot encoding 처리가 필요 없음

# model 생성
model = Sequential()

# layer 추가
# input layer
model.add(Flatten(input_shape=(norm_train_x_data.shape[1],)))

# output layer
model.add(Dense(units=10,
                activation='softmax'))

model.compile(optimizer=SGD(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model 저장 - model 구조 빼고 checkpoint 기능을 이용해서 weight, b만 저장
# 저장 위치 설정
checkpoint_path = './training_ckpt/cp.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)  # 실제 경로로 만들기
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=True,  # weight, b만 저장
                              verbose=1)


# 학습결과를 변수에 저장
history = model.fit(norm_train_x_data,
                    train_t_data,
                    epochs=100,
                    batch_size=100,
                    verbose=1,
                    validation_split=0.2,
                    callbacks=[cp_callback])

# 모델에 대한 최종 평가 진행
print(model.evaluate(norm_test_x_data, test_t_data))
#        loss               accuracy
```

    Epoch 1/100
    209/236 [=========================>....] - ETA: 0s - loss: 2.2106 - accuracy: 0.1723
    Epoch 00001: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 2.1978 - accuracy: 0.1826 - val_loss: 2.0672 - val_accuracy: 0.2893
    Epoch 2/100
    204/236 [========================>.....] - ETA: 0s - loss: 1.9896 - accuracy: 0.3729
    Epoch 00002: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.9753 - accuracy: 0.3859 - val_loss: 1.8725 - val_accuracy: 0.4869
    Epoch 3/100
    201/236 [========================>.....] - ETA: 0s - loss: 1.8122 - accuracy: 0.5345
    Epoch 00003: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.7978 - accuracy: 0.5437 - val_loss: 1.7121 - val_accuracy: 0.6005
    Epoch 4/100
    203/236 [========================>.....] - ETA: 0s - loss: 1.6599 - accuracy: 0.6212
    Epoch 00004: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.6505 - accuracy: 0.6270 - val_loss: 1.5779 - val_accuracy: 0.6658
    Epoch 5/100
    215/236 [==========================>...] - ETA: 0s - loss: 1.5312 - accuracy: 0.6781
    Epoch 00005: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.5270 - accuracy: 0.6806 - val_loss: 1.4652 - val_accuracy: 0.7049
    Epoch 6/100
    173/236 [====================>.........] - ETA: 0s - loss: 1.4384 - accuracy: 0.7075
    Epoch 00006: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.4231 - accuracy: 0.7147 - val_loss: 1.3701 - val_accuracy: 0.7281
    Epoch 7/100
    210/236 [=========================>....] - ETA: 0s - loss: 1.3369 - accuracy: 0.7401
    Epoch 00007: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.3352 - accuracy: 0.7387 - val_loss: 1.2893 - val_accuracy: 0.7463
    Epoch 8/100
    199/236 [========================>.....] - ETA: 0s - loss: 1.2656 - accuracy: 0.7527
    Epoch 00008: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.2601 - accuracy: 0.7541 - val_loss: 1.2202 - val_accuracy: 0.7587
    Epoch 9/100
    200/236 [========================>.....] - ETA: 0s - loss: 1.2026 - accuracy: 0.7624
    Epoch 00009: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.1956 - accuracy: 0.7655 - val_loss: 1.1605 - val_accuracy: 0.7723
    Epoch 10/100
    211/236 [=========================>....] - ETA: 0s - loss: 1.1432 - accuracy: 0.7755
    Epoch 00010: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.1397 - accuracy: 0.7763 - val_loss: 1.1086 - val_accuracy: 0.7828
    Epoch 11/100
    218/236 [==========================>...] - ETA: 0s - loss: 1.0937 - accuracy: 0.7840
    Epoch 00011: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.0909 - accuracy: 0.7850 - val_loss: 1.0631 - val_accuracy: 0.7901
    Epoch 12/100
    180/236 [=====================>........] - ETA: 0s - loss: 1.0537 - accuracy: 0.7910
    Epoch 00012: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.0480 - accuracy: 0.7928 - val_loss: 1.0230 - val_accuracy: 0.7957
    Epoch 13/100
    196/236 [=======================>......] - ETA: 0s - loss: 1.0126 - accuracy: 0.7977
    Epoch 00013: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 1.0099 - accuracy: 0.7983 - val_loss: 0.9873 - val_accuracy: 0.8024
    Epoch 14/100
    200/236 [========================>.....] - ETA: 0s - loss: 0.9802 - accuracy: 0.8033
    Epoch 00014: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.9760 - accuracy: 0.8038 - val_loss: 0.9554 - val_accuracy: 0.8056
    Epoch 15/100
    210/236 [=========================>....] - ETA: 0s - loss: 0.9465 - accuracy: 0.8096
    Epoch 00015: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.9455 - accuracy: 0.8092 - val_loss: 0.9267 - val_accuracy: 0.8097
    Epoch 16/100
    211/236 [=========================>....] - ETA: 0s - loss: 0.9191 - accuracy: 0.8109
    Epoch 00016: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.9181 - accuracy: 0.8129 - val_loss: 0.9007 - val_accuracy: 0.8145
    Epoch 17/100
    209/236 [=========================>....] - ETA: 0s - loss: 0.8948 - accuracy: 0.8171
    Epoch 00017: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.8931 - accuracy: 0.8169 - val_loss: 0.8771 - val_accuracy: 0.8180
    Epoch 18/100
    219/236 [==========================>...] - ETA: 0s - loss: 0.8689 - accuracy: 0.8206
    Epoch 00018: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.8704 - accuracy: 0.8203 - val_loss: 0.8555 - val_accuracy: 0.8209
    Epoch 19/100
    202/236 [========================>.....] - ETA: 0s - loss: 0.8508 - accuracy: 0.8236
    Epoch 00019: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.8496 - accuracy: 0.8238 - val_loss: 0.8357 - val_accuracy: 0.8233
    Epoch 20/100
    191/236 [=======================>......] - ETA: 0s - loss: 0.8372 - accuracy: 0.8241
    Epoch 00020: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.8305 - accuracy: 0.8262 - val_loss: 0.8175 - val_accuracy: 0.8262
    Epoch 21/100
    220/236 [==========================>...] - ETA: 0s - loss: 0.8147 - accuracy: 0.8280
    Epoch 00021: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.8128 - accuracy: 0.8281 - val_loss: 0.8007 - val_accuracy: 0.8289
    Epoch 22/100
    195/236 [=======================>......] - ETA: 0s - loss: 0.7981 - accuracy: 0.8301
    Epoch 00022: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.7965 - accuracy: 0.8305 - val_loss: 0.7851 - val_accuracy: 0.8306
    Epoch 23/100
    214/236 [==========================>...] - ETA: 0s - loss: 0.7827 - accuracy: 0.8318
    Epoch 00023: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.7813 - accuracy: 0.8326 - val_loss: 0.7707 - val_accuracy: 0.8320
    Epoch 24/100
    208/236 [=========================>....] - ETA: 0s - loss: 0.7714 - accuracy: 0.8321
    Epoch 00024: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.7672 - accuracy: 0.8342 - val_loss: 0.7572 - val_accuracy: 0.8344
    Epoch 25/100
    215/236 [==========================>...] - ETA: 0s - loss: 0.7532 - accuracy: 0.8377
    Epoch 00025: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.7540 - accuracy: 0.8363 - val_loss: 0.7446 - val_accuracy: 0.8355
    Epoch 26/100
    214/236 [==========================>...] - ETA: 0s - loss: 0.7432 - accuracy: 0.8380
    Epoch 00026: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.7416 - accuracy: 0.8385 - val_loss: 0.7327 - val_accuracy: 0.8374
    Epoch 27/100
    219/236 [==========================>...] - ETA: 0s - loss: 0.7314 - accuracy: 0.8397
    Epoch 00027: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.7300 - accuracy: 0.8403 - val_loss: 0.7217 - val_accuracy: 0.8389
    Epoch 28/100
    192/236 [=======================>......] - ETA: 0s - loss: 0.7213 - accuracy: 0.8393
    Epoch 00028: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.7191 - accuracy: 0.8415 - val_loss: 0.7112 - val_accuracy: 0.8401
    Epoch 29/100
    211/236 [=========================>....] - ETA: 0s - loss: 0.7089 - accuracy: 0.8441
    Epoch 00029: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.7088 - accuracy: 0.8435 - val_loss: 0.7014 - val_accuracy: 0.8415
    Epoch 30/100
    212/236 [=========================>....] - ETA: 0s - loss: 0.6992 - accuracy: 0.8444
    Epoch 00030: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6991 - accuracy: 0.8441 - val_loss: 0.6920 - val_accuracy: 0.8435
    Epoch 31/100
    157/236 [==================>...........] - ETA: 0s - loss: 0.6890 - accuracy: 0.8472
    Epoch 00031: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6899 - accuracy: 0.8459 - val_loss: 0.6832 - val_accuracy: 0.8440
    Epoch 32/100
    208/236 [=========================>....] - ETA: 0s - loss: 0.6824 - accuracy: 0.8469
    Epoch 00032: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6811 - accuracy: 0.8471 - val_loss: 0.6749 - val_accuracy: 0.8446
    Epoch 33/100
    217/236 [==========================>...] - ETA: 0s - loss: 0.6732 - accuracy: 0.8481
    Epoch 00033: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6729 - accuracy: 0.8479 - val_loss: 0.6670 - val_accuracy: 0.8464
    Epoch 34/100
    199/236 [========================>.....] - ETA: 0s - loss: 0.6636 - accuracy: 0.8498
    Epoch 00034: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6650 - accuracy: 0.8493 - val_loss: 0.6594 - val_accuracy: 0.8488
    Epoch 35/100
    204/236 [========================>.....] - ETA: 0s - loss: 0.6573 - accuracy: 0.8511
    Epoch 00035: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6575 - accuracy: 0.8503 - val_loss: 0.6522 - val_accuracy: 0.8505
    Epoch 36/100
    207/236 [=========================>....] - ETA: 0s - loss: 0.6511 - accuracy: 0.8518
    Epoch 00036: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6503 - accuracy: 0.8512 - val_loss: 0.6453 - val_accuracy: 0.8526
    Epoch 37/100
    212/236 [=========================>....] - ETA: 0s - loss: 0.6434 - accuracy: 0.8528
    Epoch 00037: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6435 - accuracy: 0.8524 - val_loss: 0.6388 - val_accuracy: 0.8522
    Epoch 38/100
    214/236 [==========================>...] - ETA: 0s - loss: 0.6360 - accuracy: 0.8536
    Epoch 00038: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6370 - accuracy: 0.8530 - val_loss: 0.6326 - val_accuracy: 0.8532
    Epoch 39/100
    202/236 [========================>.....] - ETA: 0s - loss: 0.6303 - accuracy: 0.8536
    Epoch 00039: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6307 - accuracy: 0.8537 - val_loss: 0.6266 - val_accuracy: 0.8536
    Epoch 40/100
    174/236 [=====================>........] - ETA: 0s - loss: 0.6279 - accuracy: 0.8528
    Epoch 00040: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6247 - accuracy: 0.8547 - val_loss: 0.6208 - val_accuracy: 0.8536
    Epoch 41/100
    185/236 [======================>.......] - ETA: 0s - loss: 0.6194 - accuracy: 0.8550
    Epoch 00041: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6190 - accuracy: 0.8558 - val_loss: 0.6153 - val_accuracy: 0.8541
    Epoch 42/100
    208/236 [=========================>....] - ETA: 0s - loss: 0.6123 - accuracy: 0.8578
    Epoch 00042: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6134 - accuracy: 0.8565 - val_loss: 0.6100 - val_accuracy: 0.8551
    Epoch 43/100
    206/236 [=========================>....] - ETA: 0s - loss: 0.6084 - accuracy: 0.8567
    Epoch 00043: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6081 - accuracy: 0.8571 - val_loss: 0.6049 - val_accuracy: 0.8553
    Epoch 44/100
    212/236 [=========================>....] - ETA: 0s - loss: 0.6031 - accuracy: 0.8578
    Epoch 00044: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.6030 - accuracy: 0.8579 - val_loss: 0.6000 - val_accuracy: 0.8573
    Epoch 45/100
    203/236 [========================>.....] - ETA: 0s - loss: 0.5945 - accuracy: 0.8615
    Epoch 00045: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5981 - accuracy: 0.8589 - val_loss: 0.5953 - val_accuracy: 0.8578
    Epoch 46/100
    200/236 [========================>.....] - ETA: 0s - loss: 0.5962 - accuracy: 0.8590
    Epoch 00046: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5934 - accuracy: 0.8596 - val_loss: 0.5907 - val_accuracy: 0.8583
    Epoch 47/100
    215/236 [==========================>...] - ETA: 0s - loss: 0.5875 - accuracy: 0.8611
    Epoch 00047: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5888 - accuracy: 0.8605 - val_loss: 0.5863 - val_accuracy: 0.8588
    Epoch 48/100
    157/236 [==================>...........] - ETA: 0s - loss: 0.5806 - accuracy: 0.8641
    Epoch 00048: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5844 - accuracy: 0.8616 - val_loss: 0.5821 - val_accuracy: 0.8600
    Epoch 49/100
    194/236 [=======================>......] - ETA: 0s - loss: 0.5785 - accuracy: 0.8618
    Epoch 00049: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5801 - accuracy: 0.8619 - val_loss: 0.5781 - val_accuracy: 0.8611
    Epoch 50/100
    202/236 [========================>.....] - ETA: 0s - loss: 0.5775 - accuracy: 0.8622
    Epoch 00050: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5760 - accuracy: 0.8627 - val_loss: 0.5741 - val_accuracy: 0.8617
    Epoch 51/100
    199/236 [========================>.....] - ETA: 0s - loss: 0.5697 - accuracy: 0.8646
    Epoch 00051: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5720 - accuracy: 0.8634 - val_loss: 0.5703 - val_accuracy: 0.8624
    Epoch 52/100
    214/236 [==========================>...] - ETA: 0s - loss: 0.5689 - accuracy: 0.8639
    Epoch 00052: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5682 - accuracy: 0.8641 - val_loss: 0.5666 - val_accuracy: 0.8636
    Epoch 53/100
    214/236 [==========================>...] - ETA: 0s - loss: 0.5643 - accuracy: 0.8643
    Epoch 00053: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5644 - accuracy: 0.8646 - val_loss: 0.5630 - val_accuracy: 0.8645
    Epoch 54/100
    206/236 [=========================>....] - ETA: 0s - loss: 0.5592 - accuracy: 0.8655
    Epoch 00054: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5608 - accuracy: 0.8653 - val_loss: 0.5596 - val_accuracy: 0.8653
    Epoch 55/100
    209/236 [=========================>....] - ETA: 0s - loss: 0.5589 - accuracy: 0.8642
    Epoch 00055: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5573 - accuracy: 0.8657 - val_loss: 0.5562 - val_accuracy: 0.8662
    Epoch 56/100
    209/236 [=========================>....] - ETA: 0s - loss: 0.5522 - accuracy: 0.8670
    Epoch 00056: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5539 - accuracy: 0.8660 - val_loss: 0.5529 - val_accuracy: 0.8672
    Epoch 57/100
    184/236 [======================>.......] - ETA: 0s - loss: 0.5498 - accuracy: 0.8669
    Epoch 00057: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5506 - accuracy: 0.8667 - val_loss: 0.5497 - val_accuracy: 0.8679
    Epoch 58/100
    183/236 [======================>.......] - ETA: 0s - loss: 0.5470 - accuracy: 0.8669
    Epoch 00058: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5474 - accuracy: 0.8671 - val_loss: 0.5467 - val_accuracy: 0.8687
    Epoch 59/100
    186/236 [======================>.......] - ETA: 0s - loss: 0.5444 - accuracy: 0.8678
    Epoch 00059: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5442 - accuracy: 0.8675 - val_loss: 0.5437 - val_accuracy: 0.8687
    Epoch 60/100
    188/236 [======================>.......] - ETA: 0s - loss: 0.5456 - accuracy: 0.8670
    Epoch 00060: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5412 - accuracy: 0.8676 - val_loss: 0.5408 - val_accuracy: 0.8685
    Epoch 61/100
    211/236 [=========================>....] - ETA: 0s - loss: 0.5380 - accuracy: 0.8679
    Epoch 00061: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5382 - accuracy: 0.8682 - val_loss: 0.5379 - val_accuracy: 0.8690
    Epoch 62/100
    217/236 [==========================>...] - ETA: 0s - loss: 0.5334 - accuracy: 0.8689
    Epoch 00062: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 979us/step - loss: 0.5354 - accuracy: 0.8687 - val_loss: 0.5352 - val_accuracy: 0.8694
    Epoch 63/100
    199/236 [========================>.....] - ETA: 0s - loss: 0.5305 - accuracy: 0.8687
    Epoch 00063: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5326 - accuracy: 0.8687 - val_loss: 0.5325 - val_accuracy: 0.8702
    Epoch 64/100
    210/236 [=========================>....] - ETA: 0s - loss: 0.5287 - accuracy: 0.8696
    Epoch 00064: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5298 - accuracy: 0.8696 - val_loss: 0.5299 - val_accuracy: 0.8704
    Epoch 65/100
    219/236 [==========================>...] - ETA: 0s - loss: 0.5292 - accuracy: 0.8689
    Epoch 00065: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5272 - accuracy: 0.8701 - val_loss: 0.5274 - val_accuracy: 0.8704
    Epoch 66/100
    215/236 [==========================>...] - ETA: 0s - loss: 0.5236 - accuracy: 0.8712
    Epoch 00066: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5246 - accuracy: 0.8705 - val_loss: 0.5249 - val_accuracy: 0.8702
    Epoch 67/100
    186/236 [======================>.......] - ETA: 0s - loss: 0.5198 - accuracy: 0.8728
    Epoch 00067: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5221 - accuracy: 0.8710 - val_loss: 0.5225 - val_accuracy: 0.8706
    Epoch 68/100
    187/236 [======================>.......] - ETA: 0s - loss: 0.5167 - accuracy: 0.8721
    Epoch 00068: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5196 - accuracy: 0.8716 - val_loss: 0.5201 - val_accuracy: 0.8716
    Epoch 69/100
    210/236 [=========================>....] - ETA: 0s - loss: 0.5165 - accuracy: 0.8711
    Epoch 00069: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5172 - accuracy: 0.8720 - val_loss: 0.5178 - val_accuracy: 0.8728
    Epoch 70/100
    214/236 [==========================>...] - ETA: 0s - loss: 0.5137 - accuracy: 0.8732
    Epoch 00070: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5148 - accuracy: 0.8722 - val_loss: 0.5156 - val_accuracy: 0.8724
    Epoch 71/100
    218/236 [==========================>...] - ETA: 0s - loss: 0.5137 - accuracy: 0.8717
    Epoch 00071: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5125 - accuracy: 0.8723 - val_loss: 0.5134 - val_accuracy: 0.8730
    Epoch 72/100
    212/236 [=========================>....] - ETA: 0s - loss: 0.5118 - accuracy: 0.8722
    Epoch 00072: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5103 - accuracy: 0.8726 - val_loss: 0.5113 - val_accuracy: 0.8735
    Epoch 73/100
    219/236 [==========================>...] - ETA: 0s - loss: 0.5066 - accuracy: 0.8733
    Epoch 00073: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5081 - accuracy: 0.8730 - val_loss: 0.5092 - val_accuracy: 0.8738
    Epoch 74/100
    211/236 [=========================>....] - ETA: 0s - loss: 0.5053 - accuracy: 0.8732
    Epoch 00074: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5060 - accuracy: 0.8736 - val_loss: 0.5071 - val_accuracy: 0.8733
    Epoch 75/100
    191/236 [=======================>......] - ETA: 0s - loss: 0.5080 - accuracy: 0.8727
    Epoch 00075: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5039 - accuracy: 0.8736 - val_loss: 0.5051 - val_accuracy: 0.8735
    Epoch 76/100
    216/236 [==========================>...] - ETA: 0s - loss: 0.5013 - accuracy: 0.8749
    Epoch 00076: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.5018 - accuracy: 0.8741 - val_loss: 0.5032 - val_accuracy: 0.8743
    Epoch 77/100
    198/236 [========================>.....] - ETA: 0s - loss: 0.5000 - accuracy: 0.8744
    Epoch 00077: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4998 - accuracy: 0.8748 - val_loss: 0.5012 - val_accuracy: 0.8740
    Epoch 78/100
    212/236 [=========================>....] - ETA: 0s - loss: 0.4955 - accuracy: 0.8758
    Epoch 00078: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4978 - accuracy: 0.8751 - val_loss: 0.4994 - val_accuracy: 0.8741
    Epoch 79/100
    219/236 [==========================>...] - ETA: 0s - loss: 0.4951 - accuracy: 0.8763
    Epoch 00079: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 979us/step - loss: 0.4959 - accuracy: 0.8756 - val_loss: 0.4975 - val_accuracy: 0.8741
    Epoch 80/100
    205/236 [=========================>....] - ETA: 0s - loss: 0.4922 - accuracy: 0.8767
    Epoch 00080: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4940 - accuracy: 0.8760 - val_loss: 0.4957 - val_accuracy: 0.8743
    Epoch 81/100
    218/236 [==========================>...] - ETA: 0s - loss: 0.4904 - accuracy: 0.8781
    Epoch 00081: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 943us/step - loss: 0.4922 - accuracy: 0.8767 - val_loss: 0.4939 - val_accuracy: 0.8750
    Epoch 82/100
    212/236 [=========================>....] - ETA: 0s - loss: 0.4920 - accuracy: 0.8761
    Epoch 00082: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4904 - accuracy: 0.8770 - val_loss: 0.4922 - val_accuracy: 0.8752
    Epoch 83/100
    216/236 [==========================>...] - ETA: 0s - loss: 0.4886 - accuracy: 0.8764
    Epoch 00083: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4886 - accuracy: 0.8770 - val_loss: 0.4905 - val_accuracy: 0.8752
    Epoch 84/100
    215/236 [==========================>...] - ETA: 0s - loss: 0.4866 - accuracy: 0.8769
    Epoch 00084: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4868 - accuracy: 0.8773 - val_loss: 0.4888 - val_accuracy: 0.8753
    Epoch 85/100
    218/236 [==========================>...] - ETA: 0s - loss: 0.4831 - accuracy: 0.8783
    Epoch 00085: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4851 - accuracy: 0.8776 - val_loss: 0.4872 - val_accuracy: 0.8755
    Epoch 86/100
    199/236 [========================>.....] - ETA: 0s - loss: 0.4862 - accuracy: 0.8769
    Epoch 00086: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4835 - accuracy: 0.8779 - val_loss: 0.4856 - val_accuracy: 0.8762
    Epoch 87/100
    186/236 [======================>.......] - ETA: 0s - loss: 0.4809 - accuracy: 0.8800
    Epoch 00087: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4818 - accuracy: 0.8779 - val_loss: 0.4841 - val_accuracy: 0.8767
    Epoch 88/100
    214/236 [==========================>...] - ETA: 0s - loss: 0.4798 - accuracy: 0.8779
    Epoch 00088: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4802 - accuracy: 0.8779 - val_loss: 0.4826 - val_accuracy: 0.8772
    Epoch 89/100
    216/236 [==========================>...] - ETA: 0s - loss: 0.4810 - accuracy: 0.8775
    Epoch 00089: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4786 - accuracy: 0.8784 - val_loss: 0.4811 - val_accuracy: 0.8776
    Epoch 90/100
    218/236 [==========================>...] - ETA: 0s - loss: 0.4790 - accuracy: 0.8780
    Epoch 00090: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4771 - accuracy: 0.8784 - val_loss: 0.4796 - val_accuracy: 0.8777
    Epoch 91/100
    209/236 [=========================>....] - ETA: 0s - loss: 0.4749 - accuracy: 0.8789
    Epoch 00091: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4756 - accuracy: 0.8788 - val_loss: 0.4781 - val_accuracy: 0.8779
    Epoch 92/100
    215/236 [==========================>...] - ETA: 0s - loss: 0.4748 - accuracy: 0.8783
    Epoch 00092: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4741 - accuracy: 0.8790 - val_loss: 0.4767 - val_accuracy: 0.8781
    Epoch 93/100
    199/236 [========================>.....] - ETA: 0s - loss: 0.4696 - accuracy: 0.8804
    Epoch 00093: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4726 - accuracy: 0.8794 - val_loss: 0.4753 - val_accuracy: 0.8781
    Epoch 94/100
    205/236 [=========================>....] - ETA: 0s - loss: 0.4710 - accuracy: 0.8795
    Epoch 00094: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4711 - accuracy: 0.8793 - val_loss: 0.4739 - val_accuracy: 0.8782
    Epoch 95/100
    192/236 [=======================>......] - ETA: 0s - loss: 0.4711 - accuracy: 0.8799
    Epoch 00095: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4697 - accuracy: 0.8795 - val_loss: 0.4726 - val_accuracy: 0.8782
    Epoch 96/100
    183/236 [======================>.......] - ETA: 0s - loss: 0.4675 - accuracy: 0.8801
    Epoch 00096: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4683 - accuracy: 0.8801 - val_loss: 0.4712 - val_accuracy: 0.8784
    Epoch 97/100
    201/236 [========================>.....] - ETA: 0s - loss: 0.4677 - accuracy: 0.8799
    Epoch 00097: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4669 - accuracy: 0.8799 - val_loss: 0.4700 - val_accuracy: 0.8786
    Epoch 98/100
    204/236 [========================>.....] - ETA: 0s - loss: 0.4675 - accuracy: 0.8803
    Epoch 00098: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4656 - accuracy: 0.8804 - val_loss: 0.4687 - val_accuracy: 0.8789
    Epoch 99/100
    201/236 [========================>.....] - ETA: 0s - loss: 0.4625 - accuracy: 0.8810
    Epoch 00099: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4643 - accuracy: 0.8808 - val_loss: 0.4674 - val_accuracy: 0.8794
    Epoch 100/100
    193/236 [=======================>......] - ETA: 0s - loss: 0.4636 - accuracy: 0.8818
    Epoch 00100: saving model to ./training_ckpt\cp.ckpt
    236/236 [==============================] - 0s 1ms/step - loss: 0.4630 - accuracy: 0.8810 - val_loss: 0.4662 - val_accuracy: 0.8794
    394/394 [==============================] - 0s 340us/step - loss: 0.4770 - accuracy: 0.8762
    [0.47703927755355835, 0.8761904835700989]
    

### 저장 후 불러서 다시 사용하기
- 학습하지 않은 상태로 evaluation을 진행한 후의 평가 결과와
- ckeckpoint 파일을 로드해서 model을 재설정한 후의 평가 결과 비교


```python
%reset

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential      # keras model
from tensorflow.keras.layers import Flatten, Dense  # Flatten(Input Layer)
                                                     # Dense(Output Layer)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint

# Raw Data Loading
df = pd.read_csv('./data/kaggle/mnist/train.csv')

train_x_data, test_x_data, train_t_data, test_t_data = \
train_test_split(df.drop('label', axis=1, inplace=False),
                 df['label'],
                 test_size=0.3,
                 random_state=1,
                 stratify=df['label'])

# 정규화
scaler = MinMaxScaler()
scaler.fit(train_x_data)

norm_train_x_data = scaler.transform(train_x_data)
norm_test_x_data = scaler.transform(test_x_data)

# model 생성
model = Sequential()

# layer 추가
# input layer
model.add(Flatten(input_shape=(norm_train_x_data.shape[1],)))

# output layer
model.add(Dense(units=10,
                activation='softmax'))

model.compile(optimizer=SGD(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 학습을 진행하지 않고 최종 평가 진행
print(model.evaluate(norm_test_x_data, test_t_data))
#        loss               accuracy
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y
    394/394 [==============================] - 0s 348us/step - loss: 2.4473 - accuracy: 0.0905
    [2.447295665740967, 0.09047619253396988]
    


```python
# checkpoint 파일에 있는 weight를 load한 후 evaluation 진행

checkpoint_path = './training_ckpt/cp.ckpt'
model.load_weights(checkpoint_path)
print(model.evaluate(norm_test_x_data, test_t_data))
```

    394/394 [==============================] - 0s 340us/step - loss: 0.4770 - accuracy: 0.8762
    [0.47703927755355835, 0.8761904835700989]
    
