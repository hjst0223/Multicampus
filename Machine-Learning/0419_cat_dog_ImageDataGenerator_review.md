# Keras가 제공하는 ImageDataGenerator 사용하기 🐾


```python
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = './data/kaggle/cat_dog_full/train'  # cats folder, dogs folder
valid_dir = './data/kaggle/cat_dog_full/validation'

# ImageDataGenerator 생성
train_datagen = ImageDataGenerator(rescale=1/255)       # 정규화 포함
validation_datagen = ImageDataGenerator(rescale=1/255)  # 정규화 포함

train_generator = train_datagen.flow_from_directory(
    train_dir,  # target directory (어떤 folder로부터 가져올지)
    classes=['cats', 'dogs'],  # folder를 어떤 인덱스값으로 쓸 것인지 [0, 1]
                               # 명시하지 않을 경우 folder 순
    target_size=(150, 150),
    batch_size=20,  # 한 번에 갖고 올 데이터 수
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    valid_dir,  # target directory
    classes=['cats', 'dogs'],  # folder를 어떤 인덱스값으로 쓸 것인지 [0, 1]
                               # 명시하지 않을 경우 folder 순
    target_size=(150, 150),
    batch_size=20,  # 한 번에 갖고 올 데이터 수 (label과 상관없이 갖고 옴)
    class_mode='binary'
)

for x_data, t_data in train_generator:
    print(x_data.shape)
    print(t_data.shape)
    break
    
figure = plt.figure()
ax = []

for i in range(20):
    ax.append(figure.add_subplot(4, 5, i + 1))

for x_data, t_data in train_generator:
    print(x_data.shape)  # (20, 150, 150, 3)
    print(t_data.shape)  # idx   img_data
    
    for idx, img_data in enumerate(x_data):
        ax[idx].imshow(img_data)
        
    break
    
plt.tight_layout()
plt.show()
```

    Found 14000 images belonging to 2 classes.
    Found 6000 images belonging to 2 classes.
    (20, 150, 150, 3)
    (20,)
    (20, 150, 150, 3)
    (20,)
    


    
![png](/Machine-Learning/images/0419/output_3_1.png)
    


## feature extraction


```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop

model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(150, 150, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 activation='relu'))

model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
```


```python
model.add(Flatten())  # 3차원 => 1차원

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
    conv2d (Conv2D)              (None, 148, 148, 32)      896       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 74, 74, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 72, 72, 64)        18496     
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 70, 70, 128)       73856     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 35, 35, 128)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 33, 33, 128)       147584    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 16, 16, 128)       0         
    _________________________________________________________________
    flatten (Flatten)            (None, 32768)             0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               8388864   
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 8,629,953
    Trainable params: 8,629,953
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

history = model.fit(train_generator,  # 14000개 이미지 20개씩 뽑아 700번
                    steps_per_epoch=700,
                    epochs=30,
                    validation_data=validation_generator,  # 6000개 이미지 20개씩 뽑아 300번
                    validation_steps=300)

model.save('./data/cats_dogs_full_cnn_model.h5')

end = timer()
print(f'Time elapsed : {timedelta(seconds=end-start)}')
```

    Epoch 1/30
    700/700 [==============================] - 196s 276ms/step - loss: 0.5912 - accuracy: 0.6746 - val_loss: 0.5196 - val_accuracy: 0.7480
    Epoch 2/30
    700/700 [==============================] - 37s 53ms/step - loss: 0.4846 - accuracy: 0.7701 - val_loss: 0.4615 - val_accuracy: 0.7782
    Epoch 3/30
    700/700 [==============================] - 37s 53ms/step - loss: 0.4245 - accuracy: 0.8051 - val_loss: 0.4518 - val_accuracy: 0.7872
    Epoch 4/30
    700/700 [==============================] - 37s 53ms/step - loss: 0.3826 - accuracy: 0.8316 - val_loss: 0.4290 - val_accuracy: 0.8063
    Epoch 5/30
    700/700 [==============================] - 38s 54ms/step - loss: 0.3432 - accuracy: 0.8487 - val_loss: 0.4014 - val_accuracy: 0.8220
    Epoch 6/30
    700/700 [==============================] - 38s 55ms/step - loss: 0.2958 - accuracy: 0.8706 - val_loss: 0.4122 - val_accuracy: 0.8183
    Epoch 7/30
    700/700 [==============================] - 38s 55ms/step - loss: 0.2561 - accuracy: 0.8928 - val_loss: 0.4293 - val_accuracy: 0.8150
    Epoch 8/30
    700/700 [==============================] - 38s 55ms/step - loss: 0.2077 - accuracy: 0.9162 - val_loss: 0.4433 - val_accuracy: 0.8210
    Epoch 9/30
    700/700 [==============================] - 38s 54ms/step - loss: 0.1596 - accuracy: 0.9387 - val_loss: 0.4736 - val_accuracy: 0.8290
    Epoch 10/30
    700/700 [==============================] - 38s 54ms/step - loss: 0.1147 - accuracy: 0.9584 - val_loss: 0.5233 - val_accuracy: 0.8107
    Epoch 11/30
    700/700 [==============================] - 38s 54ms/step - loss: 0.0775 - accuracy: 0.9734 - val_loss: 0.5722 - val_accuracy: 0.8255
    Epoch 12/30
    700/700 [==============================] - 38s 54ms/step - loss: 0.0529 - accuracy: 0.9839 - val_loss: 0.6847 - val_accuracy: 0.8165
    Epoch 13/30
    700/700 [==============================] - 38s 54ms/step - loss: 0.0402 - accuracy: 0.9882 - val_loss: 0.6904 - val_accuracy: 0.8262
    Epoch 14/30
    700/700 [==============================] - 39s 55ms/step - loss: 0.0295 - accuracy: 0.9923 - val_loss: 0.7050 - val_accuracy: 0.8282
    Epoch 15/30
    700/700 [==============================] - 39s 56ms/step - loss: 0.0238 - accuracy: 0.9926 - val_loss: 0.7164 - val_accuracy: 0.8175
    Epoch 16/30
    700/700 [==============================] - 39s 56ms/step - loss: 0.0197 - accuracy: 0.9945 - val_loss: 1.0691 - val_accuracy: 0.8003
    Epoch 17/30
    700/700 [==============================] - 39s 56ms/step - loss: 0.0179 - accuracy: 0.9947 - val_loss: 0.9151 - val_accuracy: 0.8220
    Epoch 18/30
    700/700 [==============================] - 39s 56ms/step - loss: 0.0215 - accuracy: 0.9931 - val_loss: 0.8570 - val_accuracy: 0.8255
    Epoch 19/30
    700/700 [==============================] - 39s 56ms/step - loss: 0.0203 - accuracy: 0.9938 - val_loss: 0.7938 - val_accuracy: 0.8265
    Epoch 20/30
    700/700 [==============================] - 40s 57ms/step - loss: 0.0074 - accuracy: 0.9985 - val_loss: 1.0783 - val_accuracy: 0.8187
    Epoch 21/30
    700/700 [==============================] - 40s 57ms/step - loss: 0.0129 - accuracy: 0.9962 - val_loss: 0.9872 - val_accuracy: 0.8238
    Epoch 22/30
    700/700 [==============================] - 39s 56ms/step - loss: 0.0086 - accuracy: 0.9974 - val_loss: 1.0067 - val_accuracy: 0.8252
    Epoch 23/30
    700/700 [==============================] - 38s 54ms/step - loss: 6.4149e-04 - accuracy: 1.0000 - val_loss: 1.1037 - val_accuracy: 0.8282
    Epoch 24/30
    700/700 [==============================] - 38s 54ms/step - loss: 1.7508e-04 - accuracy: 1.0000 - val_loss: 1.1698 - val_accuracy: 0.8310
    Epoch 25/30
    700/700 [==============================] - 38s 55ms/step - loss: 1.0530e-04 - accuracy: 1.0000 - val_loss: 1.2191 - val_accuracy: 0.8300
    Epoch 26/30
    700/700 [==============================] - 38s 55ms/step - loss: 7.0939e-05 - accuracy: 1.0000 - val_loss: 1.2676 - val_accuracy: 0.8298
    Epoch 27/30
    700/700 [==============================] - 38s 54ms/step - loss: 4.8631e-05 - accuracy: 1.0000 - val_loss: 1.3118 - val_accuracy: 0.8297
    Epoch 28/30
    700/700 [==============================] - 39s 56ms/step - loss: 3.4565e-05 - accuracy: 1.0000 - val_loss: 1.3609 - val_accuracy: 0.8297
    Epoch 29/30
    700/700 [==============================] - 39s 56ms/step - loss: 2.3811e-05 - accuracy: 1.0000 - val_loss: 1.4095 - val_accuracy: 0.8300
    Epoch 30/30
    700/700 [==============================] - 40s 57ms/step - loss: 1.6667e-05 - accuracy: 1.0000 - val_loss: 1.4519 - val_accuracy: 0.8307
    Time elapsed : 0:21:55.544005
    


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


    
![png](/Machine-Learning/images/0419/output_9_0.png)
    

