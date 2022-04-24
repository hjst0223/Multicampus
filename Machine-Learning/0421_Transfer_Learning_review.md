# Transfer Learning
- keras는 전이학습을 위해 VGG16 제공


```python
from tensorflow.keras.applications import VGG16

model_base = VGG16(weights='imagenet',  # 학습할 이미지 dataset
                   include_top=False,   # classification 역할의 FC layer 포함 여부
                                        # False: covolution layer까지만 가져옴
                   input_shape=(150, 150, 3))

print(model_base.summary())
```

    Model: "vgg16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 150, 150, 3)]     0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
    =================================================================
    Total params: 14,714,688
    Trainable params: 14,714,688
    Non-trainable params: 0
    _________________________________________________________________
    None
    

- 내가 갖고 있는 데이터(개와 고양이)를 VGG16에 통과시켜서
- 이미지의 특성을 ndarray 형태로 저장


```python
from timeit import default_timer as timer
from datetime import timedelta
start = timer()

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = './data/kaggle/cat_dog_small'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1/255)

def extraction_feature(directory, sample_count):  # sample_count : 이미지의 개수
    features = np.zeros(shape=(sample_count, 4, 4, 512))  # VGG16의 output shape 참고
    labels = np.zeros(shape=(sample_count,))  # 1차원
    
    generator = datagen.flow_from_directory(
        directory,
        classes=['cats', 'dogs'],
        target_size=(150, 150),  # VGG16에서 설정한 input_shape 참고
        batch_size=20,
        class_mode='binary'
    )
    
    i = 0
    
    # 20개의 이미지에 대한 4차원의 pixel 데이터, 20개의 이미지에 대한 1차원의 pixel 데이터
    for x_data_batch, t_data_batch in generator:
        feature_batch = model_base.predict(x_data_batch)  # 마지막의 layer를 통과한 결과값 (4차원 ndarray)
                                                          # 20개 이미지 특성 추출
        features[i * 20:(i + 1) * 20] = feature_batch
        labels[i * 20:(i + 1) * 20] = t_data_batch
        
        i += 1
        if i * 20 >= sample_count:
            break
            
    return features, labels
    
train_features, train_labels = extraction_feature(train_dir, 2000)

valid_features, valid_labels = extraction_feature(valid_dir, 1000)
test_features, test_labels = extraction_feature(test_dir, 1000)

end = timer()
print(f'Time : {timedelta(seconds=end-start)}')
```

    Found 2000 images belonging to 2 classes.
    Found 1000 images belonging to 2 classes.
    Found 1000 images belonging to 2 classes.
    Time : 0:01:02.343814
    

- 이 ndarray를 이용해서 DNN 학습


```python
from timeit import default_timer as timer
from datetime import timedelta
start = timer()

train_x_data = np.reshape(train_features, (2000, 4 * 4 * 512))  # 2차원
train_t_data = train_labels

valid_x_data = np.reshape(valid_features, (1000, 4 * 4 * 512))  # 2차원
valid_t_data = valid_labels

test_x_data = np.reshape(test_features, (1000, 4 * 4 * 512))    # 2차원
test_t_data = test_labels

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Flatten(input_shape=(4 * 4 * 512,)))

model.add(Dense(units=256,
                activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(units=1,
                activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_x_data,
                    train_t_data,
                    epochs=30,
                    batch_size=20,
                    validation_data=(valid_x_data, valid_t_data))
                    
end = timer()
print(f'Time : {timedelta(seconds=end-start)}')
```

    Epoch 1/30
    100/100 [==============================] - 1s 5ms/step - loss: 0.4456 - accuracy: 0.7795 - val_loss: 0.2935 - val_accuracy: 0.8820
    Epoch 2/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.2933 - accuracy: 0.8770 - val_loss: 0.2640 - val_accuracy: 0.8970
    Epoch 3/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.2326 - accuracy: 0.9080 - val_loss: 0.2489 - val_accuracy: 0.8970
    Epoch 4/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.1954 - accuracy: 0.9230 - val_loss: 0.2427 - val_accuracy: 0.9020
    Epoch 5/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.1615 - accuracy: 0.9430 - val_loss: 0.2727 - val_accuracy: 0.8930
    Epoch 6/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.1516 - accuracy: 0.9410 - val_loss: 0.2620 - val_accuracy: 0.8960
    Epoch 7/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.1206 - accuracy: 0.9615 - val_loss: 0.2365 - val_accuracy: 0.9080
    Epoch 8/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.1078 - accuracy: 0.9655 - val_loss: 0.2359 - val_accuracy: 0.9110
    Epoch 9/30
    100/100 [==============================] - 1s 6ms/step - loss: 0.0932 - accuracy: 0.9710 - val_loss: 0.2424 - val_accuracy: 0.8990
    Epoch 10/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0797 - accuracy: 0.9775 - val_loss: 0.2377 - val_accuracy: 0.9060
    Epoch 11/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.0690 - accuracy: 0.9810 - val_loss: 0.2445 - val_accuracy: 0.9000
    Epoch 12/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.0638 - accuracy: 0.9845 - val_loss: 0.2471 - val_accuracy: 0.9010
    Epoch 13/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.0558 - accuracy: 0.9855 - val_loss: 0.2707 - val_accuracy: 0.8910
    Epoch 14/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.0467 - accuracy: 0.9920 - val_loss: 0.2671 - val_accuracy: 0.9010
    Epoch 15/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.0429 - accuracy: 0.9910 - val_loss: 0.3090 - val_accuracy: 0.8820
    Epoch 16/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0418 - accuracy: 0.9935 - val_loss: 0.2588 - val_accuracy: 0.9060
    Epoch 17/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.0351 - accuracy: 0.9950 - val_loss: 0.2735 - val_accuracy: 0.9030
    Epoch 18/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0304 - accuracy: 0.9980 - val_loss: 0.2679 - val_accuracy: 0.9050
    Epoch 19/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0273 - accuracy: 0.9985 - val_loss: 0.2911 - val_accuracy: 0.8960
    Epoch 20/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.0230 - accuracy: 0.9980 - val_loss: 0.2870 - val_accuracy: 0.8990
    Epoch 21/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0190 - accuracy: 0.9985 - val_loss: 0.2800 - val_accuracy: 0.9070
    Epoch 22/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0171 - accuracy: 1.0000 - val_loss: 0.2983 - val_accuracy: 0.9020
    Epoch 23/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.0158 - accuracy: 0.9995 - val_loss: 0.3005 - val_accuracy: 0.8980
    Epoch 24/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.0146 - accuracy: 0.9990 - val_loss: 0.2989 - val_accuracy: 0.8990
    Epoch 25/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0144 - accuracy: 0.9990 - val_loss: 0.3118 - val_accuracy: 0.8980
    Epoch 26/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0112 - accuracy: 0.9995 - val_loss: 0.3202 - val_accuracy: 0.8950
    Epoch 27/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0109 - accuracy: 1.0000 - val_loss: 0.3062 - val_accuracy: 0.9040
    Epoch 28/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0098 - accuracy: 1.0000 - val_loss: 0.3123 - val_accuracy: 0.9050
    Epoch 29/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0089 - accuracy: 1.0000 - val_loss: 0.3152 - val_accuracy: 0.9070
    Epoch 30/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0078 - accuracy: 1.0000 - val_loss: 0.3215 - val_accuracy: 0.9020
    Time : 0:00:14.151748
    

## history 객체를 이용해서 결과 그래프 그리기


```python
import matplotlib.pyplot as plt

train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']

train_loss = history.history['loss']
valid_loss = history.history['val_loss']

figure = plt.figure()
ax1 = figure.add_subplot(1,2,1)
ax2 = figure.add_subplot(1,2,2)

ax1.plot(train_acc, color='r', label='train accuracy')
ax1.plot(valid_acc, color='b', label='valid accuracy')
ax1.legend()

ax2.plot(train_loss, color='r', label='train loss')
ax2.plot(valid_loss, color='b', label='valid loss')
ax2.legend()

plt.tight_layout()
plt.show()
```


    
![png](/Machine-Learning/images/0421/output_7_0.png)
    


=> overfitting 발생

# 이미지 증식을 위한 전이학습


```python
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

train_dir = './data/kaggle/cat_dog_small/train'
valid_dir = './data/kaggle/cat_dog_small/validation'

train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=30,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    classes=['cats', 'dogs'],
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')  # 다중 분류일 경우 'categorical'                                             

valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    classes=['cats', 'dogs'],
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')  # 다중 분류일 경우 'categorical'                                             
```

    Found 2000 images belonging to 2 classes.
    Found 1000 images belonging to 2 classes.
    


```python
# Pretrained Network
model_base = VGG16(weights='imagenet',
                                  include_top=False,
                                  input_shape=(150, 150, 3))

model_base.trainable = False  # Convolution layer안의 parameter 동결

print(model_base.summary())
```

    Model: "vgg16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 150, 150, 3)]     0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
    =================================================================
    Total params: 14,714,688
    Trainable params: 0
    Non-trainable params: 14,714,688
    _________________________________________________________________
    None
    


```python
# model 구현

model = Sequential()

model.add(model_base)

model.add(Flatten(input_shape=(4 * 4 * 512, )))  # 4차원 -> 2차원

model.add(Dense(units=256,
                activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(units=1,
                activation='sigmoid'))

print(model.summary())
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg16 (Functional)           (None, 4, 4, 512)         14714688  
    _________________________________________________________________
    flatten (Flatten)            (None, 8192)              0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               2097408   
    _________________________________________________________________
    dropout (Dropout)            (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 16,812,353
    Trainable params: 2,097,665
    Non-trainable params: 14,714,688
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

history = model.fit(train_generator,
                    steps_per_epoch=100,
                    epochs=30,
                    validation_data=valid_generator,
                    validation_steps=50,
                    verbose=2)

model.save('./data/kaggle/cat_dog_small/transfer_learning_cnn_cat_dog_small.h5')

end = timer()
print(f'Time elapsed : {timedelta(seconds=end-start)}')
```

    Epoch 1/30
    100/100 - 19s - loss: 0.6316 - accuracy: 0.6570 - val_loss: 0.4427 - val_accuracy: 0.8130
    Epoch 2/30
    100/100 - 13s - loss: 0.5171 - accuracy: 0.7460 - val_loss: 0.3864 - val_accuracy: 0.8300
    Epoch 3/30
    100/100 - 13s - loss: 0.4771 - accuracy: 0.7700 - val_loss: 0.3365 - val_accuracy: 0.8560
    Epoch 4/30
    100/100 - 13s - loss: 0.4577 - accuracy: 0.7805 - val_loss: 0.3259 - val_accuracy: 0.8550
    Epoch 5/30
    100/100 - 13s - loss: 0.4193 - accuracy: 0.8075 - val_loss: 0.3027 - val_accuracy: 0.8630
    Epoch 6/30
    100/100 - 13s - loss: 0.4191 - accuracy: 0.8075 - val_loss: 0.2893 - val_accuracy: 0.8710
    Epoch 7/30
    100/100 - 13s - loss: 0.4161 - accuracy: 0.7980 - val_loss: 0.3043 - val_accuracy: 0.8660
    Epoch 8/30
    100/100 - 13s - loss: 0.3918 - accuracy: 0.8250 - val_loss: 0.2816 - val_accuracy: 0.8720
    Epoch 9/30
    100/100 - 13s - loss: 0.3779 - accuracy: 0.8230 - val_loss: 0.2789 - val_accuracy: 0.8800
    Epoch 10/30
    100/100 - 13s - loss: 0.3740 - accuracy: 0.8325 - val_loss: 0.2844 - val_accuracy: 0.8680
    Epoch 11/30
    100/100 - 13s - loss: 0.3486 - accuracy: 0.8485 - val_loss: 0.2716 - val_accuracy: 0.8800
    Epoch 12/30
    100/100 - 13s - loss: 0.3762 - accuracy: 0.8335 - val_loss: 0.2692 - val_accuracy: 0.8810
    Epoch 13/30
    100/100 - 13s - loss: 0.3622 - accuracy: 0.8370 - val_loss: 0.2828 - val_accuracy: 0.8700
    Epoch 14/30
    100/100 - 13s - loss: 0.3464 - accuracy: 0.8540 - val_loss: 0.2722 - val_accuracy: 0.8740
    Epoch 15/30
    100/100 - 13s - loss: 0.3442 - accuracy: 0.8525 - val_loss: 0.2732 - val_accuracy: 0.8800
    Epoch 16/30
    100/100 - 13s - loss: 0.3464 - accuracy: 0.8505 - val_loss: 0.2793 - val_accuracy: 0.8730
    Epoch 17/30
    100/100 - 13s - loss: 0.3440 - accuracy: 0.8430 - val_loss: 0.2668 - val_accuracy: 0.8790
    Epoch 18/30
    100/100 - 13s - loss: 0.3339 - accuracy: 0.8455 - val_loss: 0.2694 - val_accuracy: 0.8770
    Epoch 19/30
    100/100 - 13s - loss: 0.3429 - accuracy: 0.8440 - val_loss: 0.2665 - val_accuracy: 0.8780
    Epoch 20/30
    100/100 - 13s - loss: 0.3435 - accuracy: 0.8415 - val_loss: 0.2632 - val_accuracy: 0.8810
    Epoch 21/30
    100/100 - 13s - loss: 0.3317 - accuracy: 0.8455 - val_loss: 0.2653 - val_accuracy: 0.8730
    Epoch 22/30
    100/100 - 13s - loss: 0.3305 - accuracy: 0.8560 - val_loss: 0.2615 - val_accuracy: 0.8810
    Epoch 23/30
    100/100 - 13s - loss: 0.3098 - accuracy: 0.8670 - val_loss: 0.2571 - val_accuracy: 0.8870
    Epoch 24/30
    100/100 - 13s - loss: 0.3350 - accuracy: 0.8510 - val_loss: 0.2571 - val_accuracy: 0.8840
    Epoch 25/30
    100/100 - 13s - loss: 0.3118 - accuracy: 0.8570 - val_loss: 0.2551 - val_accuracy: 0.8820
    Epoch 26/30
    100/100 - 13s - loss: 0.3209 - accuracy: 0.8635 - val_loss: 0.2551 - val_accuracy: 0.8930
    Epoch 27/30
    100/100 - 13s - loss: 0.3325 - accuracy: 0.8475 - val_loss: 0.2629 - val_accuracy: 0.8780
    Epoch 28/30
    100/100 - 13s - loss: 0.3144 - accuracy: 0.8595 - val_loss: 0.2524 - val_accuracy: 0.8860
    Epoch 29/30
    100/100 - 13s - loss: 0.3188 - accuracy: 0.8595 - val_loss: 0.2532 - val_accuracy: 0.8920
    Epoch 30/30
    100/100 - 12s - loss: 0.3176 - accuracy: 0.8540 - val_loss: 0.2499 - val_accuracy: 0.8890
    Time elapsed : 0:06:30.662967
    

## history 객체를 이용해서 결과 그래프 그리기


```python
import matplotlib.pyplot as plt

train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']

train_loss = history.history['loss']
valid_loss = history.history['val_loss']

figure = plt.figure()
ax1 = figure.add_subplot(1,2,1)
ax2 = figure.add_subplot(1,2,2)

ax1.plot(train_acc, color='r', label='train accuracy')
ax1.plot(valid_acc, color='b', label='valid accuracy')
ax1.legend()

ax2.plot(train_loss, color='r', label='train loss')
ax2.plot(valid_loss, color='b', label='valid loss')
ax2.legend()

plt.tight_layout()
plt.show()
```


    
![png](/Machine-Learning/images/0421/output_16_0.png)
    


### epochs=100


```python
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

from timeit import default_timer as timer
from datetime import timedelta
start = timer()

history = model.fit(train_generator,
                    steps_per_epoch=100,
                    epochs=100,
                    validation_data=valid_generator,
                    validation_steps=50,
                    verbose=2)

model.save('./data/kaggle/cat_dog_small/transfer_learning_cnn_cat_dog_small_epochs100.h5')

end = timer()
print(f'Time elapsed : {timedelta(seconds=end-start)}')
```

    Epoch 1/100
    100/100 - 14s - loss: 0.3316 - accuracy: 0.8470 - val_loss: 0.2481 - val_accuracy: 0.8920
    Epoch 2/100
    100/100 - 13s - loss: 0.3190 - accuracy: 0.8580 - val_loss: 0.2620 - val_accuracy: 0.8820
    Epoch 3/100
    100/100 - 13s - loss: 0.2868 - accuracy: 0.8765 - val_loss: 0.2482 - val_accuracy: 0.8950
    Epoch 4/100
    100/100 - 13s - loss: 0.2992 - accuracy: 0.8630 - val_loss: 0.2519 - val_accuracy: 0.8870
    Epoch 5/100
    100/100 - 13s - loss: 0.2990 - accuracy: 0.8705 - val_loss: 0.2581 - val_accuracy: 0.8840
    Epoch 6/100
    100/100 - 13s - loss: 0.3010 - accuracy: 0.8605 - val_loss: 0.2542 - val_accuracy: 0.8850
    Epoch 7/100
    100/100 - 13s - loss: 0.2998 - accuracy: 0.8680 - val_loss: 0.2549 - val_accuracy: 0.8900
    Epoch 8/100
    100/100 - 13s - loss: 0.2969 - accuracy: 0.8635 - val_loss: 0.2547 - val_accuracy: 0.8840
    Epoch 9/100
    100/100 - 13s - loss: 0.2774 - accuracy: 0.8820 - val_loss: 0.2550 - val_accuracy: 0.8830
    Epoch 10/100
    100/100 - 13s - loss: 0.2787 - accuracy: 0.8760 - val_loss: 0.2688 - val_accuracy: 0.8850
    Epoch 11/100
    100/100 - 13s - loss: 0.3108 - accuracy: 0.8635 - val_loss: 0.2623 - val_accuracy: 0.8760
    Epoch 12/100
    100/100 - 13s - loss: 0.2826 - accuracy: 0.8825 - val_loss: 0.2606 - val_accuracy: 0.8920
    Epoch 13/100
    100/100 - 13s - loss: 0.3033 - accuracy: 0.8675 - val_loss: 0.2661 - val_accuracy: 0.8820
    Epoch 14/100
    100/100 - 13s - loss: 0.2866 - accuracy: 0.8705 - val_loss: 0.2524 - val_accuracy: 0.8880
    Epoch 15/100
    100/100 - 13s - loss: 0.2910 - accuracy: 0.8755 - val_loss: 0.2509 - val_accuracy: 0.8830
    Epoch 16/100
    100/100 - 13s - loss: 0.2872 - accuracy: 0.8740 - val_loss: 0.2831 - val_accuracy: 0.8770
    Epoch 17/100
    100/100 - 13s - loss: 0.2984 - accuracy: 0.8680 - val_loss: 0.2598 - val_accuracy: 0.8860
    Epoch 18/100
    100/100 - 13s - loss: 0.2686 - accuracy: 0.8815 - val_loss: 0.2747 - val_accuracy: 0.8810
    Epoch 19/100
    100/100 - 13s - loss: 0.2776 - accuracy: 0.8780 - val_loss: 0.2552 - val_accuracy: 0.8860
    Epoch 20/100
    100/100 - 13s - loss: 0.2852 - accuracy: 0.8785 - val_loss: 0.2526 - val_accuracy: 0.8850
    Epoch 21/100
    100/100 - 13s - loss: 0.2728 - accuracy: 0.8750 - val_loss: 0.2488 - val_accuracy: 0.8840
    Epoch 22/100
    100/100 - 13s - loss: 0.2673 - accuracy: 0.8815 - val_loss: 0.2610 - val_accuracy: 0.8720
    Epoch 23/100
    100/100 - 13s - loss: 0.2844 - accuracy: 0.8750 - val_loss: 0.2490 - val_accuracy: 0.8880
    Epoch 24/100
    100/100 - 13s - loss: 0.2634 - accuracy: 0.8910 - val_loss: 0.2539 - val_accuracy: 0.8810
    Epoch 25/100
    100/100 - 13s - loss: 0.2721 - accuracy: 0.8735 - val_loss: 0.2529 - val_accuracy: 0.8850
    Epoch 26/100
    100/100 - 13s - loss: 0.2678 - accuracy: 0.8855 - val_loss: 0.2495 - val_accuracy: 0.8860
    Epoch 27/100
    100/100 - 13s - loss: 0.2650 - accuracy: 0.8845 - val_loss: 0.2473 - val_accuracy: 0.8820
    Epoch 28/100
    100/100 - 13s - loss: 0.2569 - accuracy: 0.8910 - val_loss: 0.2599 - val_accuracy: 0.8830
    Epoch 29/100
    100/100 - 13s - loss: 0.2730 - accuracy: 0.8830 - val_loss: 0.2698 - val_accuracy: 0.8750
    Epoch 30/100
    100/100 - 13s - loss: 0.2531 - accuracy: 0.8890 - val_loss: 0.2625 - val_accuracy: 0.8830
    Epoch 31/100
    100/100 - 13s - loss: 0.2493 - accuracy: 0.8960 - val_loss: 0.2583 - val_accuracy: 0.8820
    Epoch 32/100
    100/100 - 13s - loss: 0.2617 - accuracy: 0.8885 - val_loss: 0.2514 - val_accuracy: 0.8820
    Epoch 33/100
    100/100 - 13s - loss: 0.2724 - accuracy: 0.8820 - val_loss: 0.2745 - val_accuracy: 0.8800
    Epoch 34/100
    100/100 - 13s - loss: 0.2513 - accuracy: 0.8930 - val_loss: 0.2547 - val_accuracy: 0.8760
    Epoch 35/100
    100/100 - 13s - loss: 0.2519 - accuracy: 0.8930 - val_loss: 0.2709 - val_accuracy: 0.8880
    Epoch 36/100
    100/100 - 13s - loss: 0.2436 - accuracy: 0.8960 - val_loss: 0.2670 - val_accuracy: 0.8760
    Epoch 37/100
    100/100 - 13s - loss: 0.2635 - accuracy: 0.8895 - val_loss: 0.2473 - val_accuracy: 0.8880
    Epoch 38/100
    100/100 - 13s - loss: 0.2237 - accuracy: 0.9040 - val_loss: 0.2619 - val_accuracy: 0.8880
    Epoch 39/100
    100/100 - 13s - loss: 0.2426 - accuracy: 0.8890 - val_loss: 0.2562 - val_accuracy: 0.8890
    Epoch 40/100
    100/100 - 13s - loss: 0.2503 - accuracy: 0.9015 - val_loss: 0.2583 - val_accuracy: 0.8850
    Epoch 41/100
    100/100 - 13s - loss: 0.2370 - accuracy: 0.8950 - val_loss: 0.2627 - val_accuracy: 0.8840
    Epoch 42/100
    100/100 - 13s - loss: 0.2430 - accuracy: 0.8980 - val_loss: 0.2579 - val_accuracy: 0.8830
    Epoch 43/100
    100/100 - 13s - loss: 0.2436 - accuracy: 0.8910 - val_loss: 0.2628 - val_accuracy: 0.8840
    Epoch 44/100
    100/100 - 13s - loss: 0.2526 - accuracy: 0.8925 - val_loss: 0.2622 - val_accuracy: 0.8820
    Epoch 45/100
    100/100 - 13s - loss: 0.2376 - accuracy: 0.8945 - val_loss: 0.2641 - val_accuracy: 0.8820
    Epoch 46/100
    100/100 - 13s - loss: 0.2385 - accuracy: 0.8920 - val_loss: 0.2519 - val_accuracy: 0.8950
    Epoch 47/100
    100/100 - 13s - loss: 0.2494 - accuracy: 0.8875 - val_loss: 0.2578 - val_accuracy: 0.8910
    Epoch 48/100
    100/100 - 13s - loss: 0.2241 - accuracy: 0.9040 - val_loss: 0.2531 - val_accuracy: 0.8910
    Epoch 49/100
    100/100 - 13s - loss: 0.2237 - accuracy: 0.9005 - val_loss: 0.2575 - val_accuracy: 0.8820
    Epoch 50/100
    100/100 - 13s - loss: 0.2164 - accuracy: 0.9070 - val_loss: 0.2647 - val_accuracy: 0.8830
    Epoch 51/100
    100/100 - 13s - loss: 0.2275 - accuracy: 0.9100 - val_loss: 0.2654 - val_accuracy: 0.8810
    Epoch 52/100
    100/100 - 13s - loss: 0.2290 - accuracy: 0.9030 - val_loss: 0.2645 - val_accuracy: 0.8830
    Epoch 53/100
    100/100 - 13s - loss: 0.2239 - accuracy: 0.9055 - val_loss: 0.2581 - val_accuracy: 0.8880
    Epoch 54/100
    100/100 - 13s - loss: 0.2257 - accuracy: 0.8970 - val_loss: 0.2607 - val_accuracy: 0.8930
    Epoch 55/100
    100/100 - 13s - loss: 0.2221 - accuracy: 0.9090 - val_loss: 0.2596 - val_accuracy: 0.8910
    Epoch 56/100
    100/100 - 13s - loss: 0.2360 - accuracy: 0.8990 - val_loss: 0.2599 - val_accuracy: 0.8840
    Epoch 57/100
    100/100 - 13s - loss: 0.2058 - accuracy: 0.9170 - val_loss: 0.2698 - val_accuracy: 0.8790
    Epoch 58/100
    100/100 - 13s - loss: 0.2159 - accuracy: 0.9080 - val_loss: 0.2736 - val_accuracy: 0.8820
    Epoch 59/100
    100/100 - 13s - loss: 0.2288 - accuracy: 0.9025 - val_loss: 0.2669 - val_accuracy: 0.8800
    Epoch 60/100
    100/100 - 13s - loss: 0.2222 - accuracy: 0.9045 - val_loss: 0.2590 - val_accuracy: 0.8920
    Epoch 61/100
    100/100 - 13s - loss: 0.2278 - accuracy: 0.9015 - val_loss: 0.2621 - val_accuracy: 0.8770
    Epoch 62/100
    100/100 - 13s - loss: 0.2235 - accuracy: 0.9075 - val_loss: 0.2591 - val_accuracy: 0.8880
    Epoch 63/100
    100/100 - 13s - loss: 0.2219 - accuracy: 0.9105 - val_loss: 0.2516 - val_accuracy: 0.8870
    Epoch 64/100
    100/100 - 13s - loss: 0.2133 - accuracy: 0.9155 - val_loss: 0.2571 - val_accuracy: 0.8950
    Epoch 65/100
    100/100 - 13s - loss: 0.2110 - accuracy: 0.9120 - val_loss: 0.2712 - val_accuracy: 0.8810
    Epoch 66/100
    100/100 - 13s - loss: 0.2272 - accuracy: 0.9115 - val_loss: 0.2545 - val_accuracy: 0.8890
    Epoch 67/100
    100/100 - 13s - loss: 0.2162 - accuracy: 0.9115 - val_loss: 0.2500 - val_accuracy: 0.8940
    Epoch 68/100
    100/100 - 13s - loss: 0.2203 - accuracy: 0.9090 - val_loss: 0.2447 - val_accuracy: 0.8880
    Epoch 69/100
    100/100 - 13s - loss: 0.2160 - accuracy: 0.9120 - val_loss: 0.2521 - val_accuracy: 0.8940
    Epoch 70/100
    100/100 - 13s - loss: 0.2181 - accuracy: 0.9090 - val_loss: 0.2466 - val_accuracy: 0.8940
    Epoch 71/100
    100/100 - 13s - loss: 0.2206 - accuracy: 0.9010 - val_loss: 0.2539 - val_accuracy: 0.8870
    Epoch 72/100
    100/100 - 13s - loss: 0.2277 - accuracy: 0.9050 - val_loss: 0.2508 - val_accuracy: 0.8920
    Epoch 73/100
    100/100 - 14s - loss: 0.2140 - accuracy: 0.9125 - val_loss: 0.2620 - val_accuracy: 0.8860
    Epoch 74/100
    100/100 - 14s - loss: 0.2040 - accuracy: 0.9120 - val_loss: 0.2675 - val_accuracy: 0.8780
    Epoch 75/100
    100/100 - 14s - loss: 0.2150 - accuracy: 0.9155 - val_loss: 0.2551 - val_accuracy: 0.8940
    Epoch 76/100
    100/100 - 14s - loss: 0.2152 - accuracy: 0.9015 - val_loss: 0.2563 - val_accuracy: 0.8920
    Epoch 77/100
    100/100 - 14s - loss: 0.2055 - accuracy: 0.9165 - val_loss: 0.2637 - val_accuracy: 0.8970
    Epoch 78/100
    100/100 - 14s - loss: 0.2156 - accuracy: 0.9115 - val_loss: 0.2511 - val_accuracy: 0.8920
    Epoch 79/100
    100/100 - 14s - loss: 0.2003 - accuracy: 0.9220 - val_loss: 0.2550 - val_accuracy: 0.8900
    Epoch 80/100
    100/100 - 14s - loss: 0.2035 - accuracy: 0.9140 - val_loss: 0.2620 - val_accuracy: 0.8880
    Epoch 81/100
    100/100 - 14s - loss: 0.2036 - accuracy: 0.9155 - val_loss: 0.2596 - val_accuracy: 0.9000
    Epoch 82/100
    100/100 - 14s - loss: 0.2087 - accuracy: 0.9165 - val_loss: 0.2438 - val_accuracy: 0.9040
    Epoch 83/100
    100/100 - 14s - loss: 0.1961 - accuracy: 0.9220 - val_loss: 0.2597 - val_accuracy: 0.8920
    Epoch 84/100
    100/100 - 14s - loss: 0.2189 - accuracy: 0.9050 - val_loss: 0.2551 - val_accuracy: 0.8960
    Epoch 85/100
    100/100 - 14s - loss: 0.2065 - accuracy: 0.9115 - val_loss: 0.2616 - val_accuracy: 0.8870
    Epoch 86/100
    100/100 - 14s - loss: 0.2045 - accuracy: 0.9125 - val_loss: 0.2838 - val_accuracy: 0.8710
    Epoch 87/100
    100/100 - 14s - loss: 0.1986 - accuracy: 0.9200 - val_loss: 0.2637 - val_accuracy: 0.8920
    Epoch 88/100
    100/100 - 14s - loss: 0.2003 - accuracy: 0.9170 - val_loss: 0.2724 - val_accuracy: 0.8870
    Epoch 89/100
    100/100 - 14s - loss: 0.1872 - accuracy: 0.9280 - val_loss: 0.2632 - val_accuracy: 0.8920
    Epoch 90/100
    100/100 - 14s - loss: 0.1950 - accuracy: 0.9170 - val_loss: 0.2725 - val_accuracy: 0.8850
    Epoch 91/100
    100/100 - 14s - loss: 0.1857 - accuracy: 0.9225 - val_loss: 0.2696 - val_accuracy: 0.8920
    Epoch 92/100
    100/100 - 14s - loss: 0.1925 - accuracy: 0.9145 - val_loss: 0.2742 - val_accuracy: 0.8900
    Epoch 93/100
    100/100 - 14s - loss: 0.1977 - accuracy: 0.9220 - val_loss: 0.2725 - val_accuracy: 0.8860
    Epoch 94/100
    100/100 - 14s - loss: 0.1970 - accuracy: 0.9155 - val_loss: 0.2741 - val_accuracy: 0.8870
    Epoch 95/100
    100/100 - 14s - loss: 0.1789 - accuracy: 0.9330 - val_loss: 0.2849 - val_accuracy: 0.8740
    Epoch 96/100
    100/100 - 14s - loss: 0.1885 - accuracy: 0.9220 - val_loss: 0.2758 - val_accuracy: 0.8820
    Epoch 97/100
    100/100 - 14s - loss: 0.1885 - accuracy: 0.9185 - val_loss: 0.2731 - val_accuracy: 0.8820
    Epoch 98/100
    100/100 - 14s - loss: 0.1965 - accuracy: 0.9180 - val_loss: 0.2674 - val_accuracy: 0.8840
    Epoch 99/100
    100/100 - 14s - loss: 0.1849 - accuracy: 0.9185 - val_loss: 0.2676 - val_accuracy: 0.8910
    Epoch 100/100
    100/100 - 14s - loss: 0.1934 - accuracy: 0.9205 - val_loss: 0.2738 - val_accuracy: 0.8850
    Time elapsed : 0:22:05.290999
    


```python
import matplotlib.pyplot as plt

train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']

train_loss = history.history['loss']
valid_loss = history.history['val_loss']

figure = plt.figure()
ax1 = figure.add_subplot(1,2,1)
ax2 = figure.add_subplot(1,2,2)

ax1.plot(train_acc, color='r', label='train accuracy')
ax1.plot(valid_acc, color='b', label='valid accuracy')
ax1.legend()

ax2.plot(train_loss, color='r', label='train loss')
ax2.plot(valid_loss, color='b', label='valid loss')
ax2.legend()

plt.tight_layout()
plt.show()
```


    
![png](/Machine-Learning/images/0421/output_19_0.png)
    

