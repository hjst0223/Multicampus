# 이미지 증식을 위한 전이학습 (fine tuning)


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
                                                    class_mode='binary')


# Pretrained Network
model_base = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(150, 150, 3))

model_base.trainable = False  # Convolution layer안의 parameter 동결

print(model_base.summary())


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

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

from timeit import default_timer as timer
from datetime import timedelta
start = timer()

history = model.fit(train_generator,
                    steps_per_epoch=100,
                    epochs=30,
                    validation_data=valid_generator,
                    validation_steps=50,
                    verbose=2)

end = timer()
print(f'Time elapsed : {timedelta(seconds=end-start)}')

# 여기까지 수행하면 classifier가 학습됨

model_base.trainable = True  # 동결 해제

for layer in model_base.layers:
    if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
        layer.trainable = True  # 동결 해제
    else:
        layer.trainable = False  # 동결
        
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 재학습 진행
start = timer()

history = model.fit(train_generator,
                    steps_per_epoch=100,
                    epochs=30,
                    validation_data=valid_generator,
                    validation_steps=50,
                    verbose=2)

end = timer()
print(f'Time elapsed : {timedelta(seconds=end-start)}')
```

    Found 2000 images belonging to 2 classes.
    Found 1000 images belonging to 2 classes.
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
    Epoch 1/30
    100/100 - 19s - loss: 0.6317 - accuracy: 0.6560 - val_loss: 0.4180 - val_accuracy: 0.8020
    Epoch 2/30
    100/100 - 13s - loss: 0.4943 - accuracy: 0.7515 - val_loss: 0.3485 - val_accuracy: 0.8440
    Epoch 3/30
    100/100 - 13s - loss: 0.4562 - accuracy: 0.7890 - val_loss: 0.3414 - val_accuracy: 0.8430
    Epoch 4/30
    100/100 - 14s - loss: 0.4363 - accuracy: 0.7985 - val_loss: 0.2997 - val_accuracy: 0.8730
    Epoch 5/30
    100/100 - 13s - loss: 0.4166 - accuracy: 0.8010 - val_loss: 0.2968 - val_accuracy: 0.8740
    Epoch 6/30
    100/100 - 13s - loss: 0.3953 - accuracy: 0.8185 - val_loss: 0.2831 - val_accuracy: 0.8680
    Epoch 7/30
    100/100 - 13s - loss: 0.3913 - accuracy: 0.8245 - val_loss: 0.2811 - val_accuracy: 0.8710
    Epoch 8/30
    100/100 - 14s - loss: 0.3678 - accuracy: 0.8365 - val_loss: 0.2801 - val_accuracy: 0.8710
    Epoch 9/30
    100/100 - 14s - loss: 0.3719 - accuracy: 0.8300 - val_loss: 0.2780 - val_accuracy: 0.8740
    Epoch 10/30
    100/100 - 14s - loss: 0.3806 - accuracy: 0.8280 - val_loss: 0.2746 - val_accuracy: 0.8750
    Epoch 11/30
    100/100 - 14s - loss: 0.3642 - accuracy: 0.8340 - val_loss: 0.2740 - val_accuracy: 0.8710
    Epoch 12/30
    100/100 - 13s - loss: 0.3652 - accuracy: 0.8315 - val_loss: 0.2769 - val_accuracy: 0.8690
    Epoch 13/30
    100/100 - 13s - loss: 0.3488 - accuracy: 0.8500 - val_loss: 0.2704 - val_accuracy: 0.8770
    Epoch 14/30
    100/100 - 13s - loss: 0.3506 - accuracy: 0.8460 - val_loss: 0.2706 - val_accuracy: 0.8810
    Epoch 15/30
    100/100 - 13s - loss: 0.3504 - accuracy: 0.8440 - val_loss: 0.2635 - val_accuracy: 0.8860
    Epoch 16/30
    100/100 - 13s - loss: 0.3470 - accuracy: 0.8410 - val_loss: 0.2624 - val_accuracy: 0.8820
    Epoch 17/30
    100/100 - 13s - loss: 0.3286 - accuracy: 0.8505 - val_loss: 0.2598 - val_accuracy: 0.8800
    Epoch 18/30
    100/100 - 13s - loss: 0.3439 - accuracy: 0.8490 - val_loss: 0.2864 - val_accuracy: 0.8680
    Epoch 19/30
    100/100 - 13s - loss: 0.3583 - accuracy: 0.8405 - val_loss: 0.2683 - val_accuracy: 0.8720
    Epoch 20/30
    100/100 - 13s - loss: 0.3308 - accuracy: 0.8570 - val_loss: 0.2597 - val_accuracy: 0.8790
    Epoch 21/30
    100/100 - 13s - loss: 0.3111 - accuracy: 0.8625 - val_loss: 0.2604 - val_accuracy: 0.8780
    Epoch 22/30
    100/100 - 13s - loss: 0.3195 - accuracy: 0.8615 - val_loss: 0.2827 - val_accuracy: 0.8730
    Epoch 23/30
    100/100 - 13s - loss: 0.3265 - accuracy: 0.8560 - val_loss: 0.2811 - val_accuracy: 0.8700
    Epoch 24/30
    100/100 - 13s - loss: 0.3435 - accuracy: 0.8485 - val_loss: 0.2616 - val_accuracy: 0.8820
    Epoch 25/30
    100/100 - 13s - loss: 0.3170 - accuracy: 0.8610 - val_loss: 0.2557 - val_accuracy: 0.8810
    Epoch 26/30
    100/100 - 13s - loss: 0.3135 - accuracy: 0.8585 - val_loss: 0.2596 - val_accuracy: 0.8810
    Epoch 27/30
    100/100 - 13s - loss: 0.3274 - accuracy: 0.8575 - val_loss: 0.2606 - val_accuracy: 0.8820
    Epoch 28/30
    100/100 - 13s - loss: 0.3247 - accuracy: 0.8530 - val_loss: 0.2653 - val_accuracy: 0.8740
    Epoch 29/30
    100/100 - 13s - loss: 0.2974 - accuracy: 0.8650 - val_loss: 0.2678 - val_accuracy: 0.8800
    Epoch 30/30
    100/100 - 13s - loss: 0.3175 - accuracy: 0.8675 - val_loss: 0.2557 - val_accuracy: 0.8860
    Time elapsed : 0:06:44.762736
    Epoch 1/30
    100/100 - 14s - loss: 0.3148 - accuracy: 0.8605 - val_loss: 0.2469 - val_accuracy: 0.8920
    Epoch 2/30
    100/100 - 13s - loss: 0.2711 - accuracy: 0.8855 - val_loss: 0.2350 - val_accuracy: 0.8920
    Epoch 3/30
    100/100 - 13s - loss: 0.2476 - accuracy: 0.8870 - val_loss: 0.2225 - val_accuracy: 0.8970
    Epoch 4/30
    100/100 - 13s - loss: 0.2495 - accuracy: 0.8970 - val_loss: 0.2363 - val_accuracy: 0.8950
    Epoch 5/30
    100/100 - 14s - loss: 0.2248 - accuracy: 0.9015 - val_loss: 0.2238 - val_accuracy: 0.8990
    Epoch 6/30
    100/100 - 14s - loss: 0.2097 - accuracy: 0.9100 - val_loss: 0.2268 - val_accuracy: 0.8960
    Epoch 7/30
    100/100 - 13s - loss: 0.1996 - accuracy: 0.9115 - val_loss: 0.2263 - val_accuracy: 0.9040
    Epoch 8/30
    100/100 - 13s - loss: 0.1905 - accuracy: 0.9215 - val_loss: 0.2371 - val_accuracy: 0.8990
    Epoch 9/30
    100/100 - 13s - loss: 0.1825 - accuracy: 0.9225 - val_loss: 0.2347 - val_accuracy: 0.9000
    Epoch 10/30
    100/100 - 13s - loss: 0.1686 - accuracy: 0.9290 - val_loss: 0.2303 - val_accuracy: 0.8990
    Epoch 11/30
    100/100 - 13s - loss: 0.1448 - accuracy: 0.9455 - val_loss: 0.2425 - val_accuracy: 0.9040
    Epoch 12/30
    100/100 - 13s - loss: 0.1573 - accuracy: 0.9380 - val_loss: 0.2288 - val_accuracy: 0.9100
    Epoch 13/30
    100/100 - 13s - loss: 0.1530 - accuracy: 0.9380 - val_loss: 0.2322 - val_accuracy: 0.9060
    Epoch 14/30
    100/100 - 13s - loss: 0.1434 - accuracy: 0.9340 - val_loss: 0.2409 - val_accuracy: 0.9080
    Epoch 15/30
    100/100 - 13s - loss: 0.1433 - accuracy: 0.9460 - val_loss: 0.2206 - val_accuracy: 0.9080
    Epoch 16/30
    100/100 - 13s - loss: 0.1337 - accuracy: 0.9490 - val_loss: 0.2263 - val_accuracy: 0.9090
    Epoch 17/30
    100/100 - 13s - loss: 0.1236 - accuracy: 0.9480 - val_loss: 0.2562 - val_accuracy: 0.9020
    Epoch 18/30
    100/100 - 13s - loss: 0.1189 - accuracy: 0.9525 - val_loss: 0.2258 - val_accuracy: 0.9100
    Epoch 19/30
    100/100 - 13s - loss: 0.0995 - accuracy: 0.9595 - val_loss: 0.2254 - val_accuracy: 0.9110
    Epoch 20/30
    100/100 - 13s - loss: 0.1148 - accuracy: 0.9530 - val_loss: 0.2840 - val_accuracy: 0.8930
    Epoch 21/30
    100/100 - 13s - loss: 0.1012 - accuracy: 0.9625 - val_loss: 0.2869 - val_accuracy: 0.9010
    Epoch 22/30
    100/100 - 13s - loss: 0.0938 - accuracy: 0.9665 - val_loss: 0.2519 - val_accuracy: 0.9070
    Epoch 23/30
    100/100 - 13s - loss: 0.0986 - accuracy: 0.9630 - val_loss: 0.2430 - val_accuracy: 0.9150
    Epoch 24/30
    100/100 - 13s - loss: 0.0940 - accuracy: 0.9600 - val_loss: 0.3046 - val_accuracy: 0.8950
    Epoch 25/30
    100/100 - 13s - loss: 0.0860 - accuracy: 0.9645 - val_loss: 0.2722 - val_accuracy: 0.9070
    Epoch 26/30
    100/100 - 13s - loss: 0.0878 - accuracy: 0.9695 - val_loss: 0.2560 - val_accuracy: 0.9050
    Epoch 27/30
    100/100 - 13s - loss: 0.1002 - accuracy: 0.9615 - val_loss: 0.2707 - val_accuracy: 0.9090
    Epoch 28/30
    100/100 - 13s - loss: 0.0968 - accuracy: 0.9650 - val_loss: 0.2372 - val_accuracy: 0.9160
    Epoch 29/30
    100/100 - 13s - loss: 0.0813 - accuracy: 0.9675 - val_loss: 0.2810 - val_accuracy: 0.9020
    Epoch 30/30
    100/100 - 13s - loss: 0.0865 - accuracy: 0.9710 - val_loss: 0.2624 - val_accuracy: 0.9120
    Time elapsed : 0:06:32.241047
    

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


    
![png](/Machine-Learning/images/0421/output_3_0.png)
    



```python
from tensorflow.keras.applications import VGG16

model_base = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(150, 150, 3))

print(model_base.summary())
```

    Model: "vgg16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 150, 150, 3)]     0         
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
    
