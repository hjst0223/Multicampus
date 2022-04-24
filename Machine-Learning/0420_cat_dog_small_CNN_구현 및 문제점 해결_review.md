# 데이터 나누기 🐱‍🐉
### 각각 2,000개의 (고양이와 개의) 이미지 파일을 다음과 같이 분배
- train : 1,000 
- validation : 500
- test : 500 


```python
import os, shutil

original_dataset_dir = './data/kaggle/cat_dog/train'

base_dir = 'data/kaggle/cat_dog_small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir,'train').replace('\\','/')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation').replace('\\','/')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test').replace('\\','/')
os.mkdir(test_dir)


train_cats_dir = os.path.join(train_dir,'cats').replace('\\','/')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir,'dogs').replace('\\','/')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir,'cats').replace('\\','/')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir,'dogs').replace('\\','/')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir,'cats').replace('\\','/')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir,'dogs').replace('\\','/')
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(train_cats_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(validation_cats_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(test_cats_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(train_dogs_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(validation_dogs_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(test_dogs_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)
```


```python
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = './data/kaggle/cat_dog_small/train'  # cats folder, dogs folder
valid_dir = './data/kaggle/cat_dog_small/validation'

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

    Found 2000 images belonging to 2 classes.
    Found 1000 images belonging to 2 classes.
    (20, 150, 150, 3)
    (20,)
    (20, 150, 150, 3)
    (20,)
    


    
![png](/Machine-Learning/images/0420/output_2_1.png)
    


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

history = model.fit(train_generator,  # 2000개 이미지 20개씩 뽑아 100번
                    steps_per_epoch=100,
                    epochs=30,
                    validation_data=validation_generator,  # 1000개 이미지 20개씩 뽑아 50번
                    validation_steps=50)

model.save('./data/kaggle/cats_dogs_small_cnn_model.h5')

end = timer()
print(f'Time : {timedelta(seconds=end-start)}')
```

    Epoch 1/30
    100/100 [==============================] - 9s 59ms/step - loss: 0.6957 - accuracy: 0.5020 - val_loss: 0.6870 - val_accuracy: 0.5000
    Epoch 2/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.6588 - accuracy: 0.6000 - val_loss: 0.6464 - val_accuracy: 0.6460
    Epoch 3/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.5853 - accuracy: 0.6805 - val_loss: 0.6301 - val_accuracy: 0.6350
    Epoch 4/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.5290 - accuracy: 0.7475 - val_loss: 0.5863 - val_accuracy: 0.6940
    Epoch 5/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.4940 - accuracy: 0.7620 - val_loss: 0.5757 - val_accuracy: 0.7030
    Epoch 6/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.4485 - accuracy: 0.7925 - val_loss: 0.5747 - val_accuracy: 0.7110
    Epoch 7/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4261 - accuracy: 0.8035 - val_loss: 0.6340 - val_accuracy: 0.6940
    Epoch 8/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.3825 - accuracy: 0.8310 - val_loss: 0.5991 - val_accuracy: 0.7000
    Epoch 9/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.3404 - accuracy: 0.8510 - val_loss: 0.5892 - val_accuracy: 0.7290
    Epoch 10/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.3024 - accuracy: 0.8705 - val_loss: 0.6340 - val_accuracy: 0.7160
    Epoch 11/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.2775 - accuracy: 0.8855 - val_loss: 0.6493 - val_accuracy: 0.7280
    Epoch 12/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.2335 - accuracy: 0.9120 - val_loss: 0.6926 - val_accuracy: 0.7200
    Epoch 13/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.2100 - accuracy: 0.9155 - val_loss: 0.7027 - val_accuracy: 0.7170
    Epoch 14/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.1767 - accuracy: 0.9325 - val_loss: 0.7406 - val_accuracy: 0.7290
    Epoch 15/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.1262 - accuracy: 0.9605 - val_loss: 0.8426 - val_accuracy: 0.7280
    Epoch 16/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.1011 - accuracy: 0.9745 - val_loss: 1.0127 - val_accuracy: 0.6910
    Epoch 17/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.0973 - accuracy: 0.9695 - val_loss: 0.9164 - val_accuracy: 0.7220
    Epoch 18/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.0568 - accuracy: 0.9865 - val_loss: 1.0208 - val_accuracy: 0.7120
    Epoch 19/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.0417 - accuracy: 0.9930 - val_loss: 1.1289 - val_accuracy: 0.7110
    Epoch 20/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.0298 - accuracy: 0.9955 - val_loss: 1.2485 - val_accuracy: 0.7060
    Epoch 21/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.0198 - accuracy: 0.9985 - val_loss: 1.2711 - val_accuracy: 0.7180
    Epoch 22/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.0124 - accuracy: 1.0000 - val_loss: 1.3788 - val_accuracy: 0.7270
    Epoch 23/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.0074 - accuracy: 1.0000 - val_loss: 1.4857 - val_accuracy: 0.7120
    Epoch 24/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.0054 - accuracy: 1.0000 - val_loss: 1.4867 - val_accuracy: 0.7200
    Epoch 25/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.0043 - accuracy: 1.0000 - val_loss: 1.5723 - val_accuracy: 0.7190
    Epoch 26/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.0037 - accuracy: 1.0000 - val_loss: 1.5917 - val_accuracy: 0.7280
    Epoch 27/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.0029 - accuracy: 1.0000 - val_loss: 1.6391 - val_accuracy: 0.7280
    Epoch 28/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.0022 - accuracy: 1.0000 - val_loss: 1.6716 - val_accuracy: 0.7260
    Epoch 29/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.0021 - accuracy: 1.0000 - val_loss: 1.7028 - val_accuracy: 0.7270
    Epoch 30/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 1.7292 - val_accuracy: 0.7160
    Time : 0:02:54.915463
    


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


    
![png](/Machine-Learning/images/0420/output_8_0.png)
    


## Image Augmentation (이미지 증식)


```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# train_datagen = ImageDataGenerator(rescale=1/255)
datagen = ImageDataGenerator(rotation_range=20,       # 최대 회전 각도 (0~20)
                             width_shift_range=0.1,   # 가로 방향으로 이미지 이동시키기 (0~10%)
                             height_shift_range=0.5,  # 세로 방향으로 이미지 이동시키기 (0~50%) 
                             zoom_range=0.1,          # 10%의 비율로 확대/축소
                             horizontal_flip=True,    # 좌우 반전 (True:허용)
                             vertical_flip=True,      # 상하 반전 (True:허용)
                             fill_mode='nearest')     # 이미지 보정

img = image.load_img('./data/kaggle/cat_dog_small/train/cats/cat.3.jpg',
                     target_size=(150, 150))

x = image.img_to_array(img)  # 이미지 데이터로부터 numpy(pixel) 데이터 뽑아냄
print(type(x), x.shape)  # (세로, 가로, 채널)
```

    <class 'numpy.ndarray'> (150, 150, 3)
    


```python
x = x.reshape((1, ) + x.shape)
print(x.shape)
```

    (1, 150, 150, 3)
    


```python
figure = plt.figure()
ax = []

for i in range(20):
    ax.append(figure.add_subplot(4, 5, i + 1))
    
idx = 0

# Genreator를 이용해서 데이터 가져오기 - 증식
for batch in datagen.flow(x, batch_size=1):       # x를 기반으로 데이터 1개 가져오기
    ax[idx].imshow(image.array_to_img(batch[0]))  # numpy(pixel) 데이터 => 이미지 데이터
    idx += 1
    if idx == 20:
        break
        
plt.tight_layout()
plt.show()
```


    
![png](/Machine-Learning/images/0420/output_12_0.png)
    


## 증식을 이용해서 4000개의 이미지 학습


```python
from timeit import default_timer as timer
from datetime import timedelta
start = timer()

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = './data/kaggle/cat_dog_small/train'  # cats folder, dogs folder
valid_dir = './data/kaggle/cat_dog_small/validation'

# ImageDataGenerator 생성
train_datagen = ImageDataGenerator(rescale=1/255,           # 정규화 포함
                                   rotation_range=30,       # 최대 회전 각도 (0~20)
                                   width_shift_range=0.1,   # 가로 방향으로 이미지 이동시키기 (0~10%)
                                   height_shift_range=0.1,  # 세로 방향으로 이미지 이동시키기 (0~50%) 
                                   zoom_range=0.2,          # 10%의 비율로 확대/축소
                                   horizontal_flip=True,    # 좌우 반전 (True:허용)
                                   fill_mode='nearest')     # 이미지 보정

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

model.add(Flatten())  # 3차원 => 1차원

model.add(Dense(units=256,
                activation='relu'))

model.add(Dense(units=1,
                activation='sigmoid'))

print(model.summary())

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,  # 2000개 이미지 20개씩 뽑아 100번
                    steps_per_epoch=100,
                    epochs=30,
                    validation_data=validation_generator,  # 1000개 이미지 20개씩 뽑아 50번
                    validation_steps=50)

model.save('./data/kaggle/cats_dogs_small_cnn_model_augmentation.h5')

end = timer()
print(f'Time : {timedelta(seconds=end-start)}')
```

    Found 2000 images belonging to 2 classes.
    Found 1000 images belonging to 2 classes.
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 148, 148, 32)      896       
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 74, 74, 32)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 72, 72, 64)        18496     
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 70, 70, 128)       73856     
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 35, 35, 128)       0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 33, 33, 128)       147584    
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 16, 16, 128)       0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 32768)             0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 256)               8388864   
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 8,629,953
    Trainable params: 8,629,953
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/30
    100/100 [==============================] - 13s 125ms/step - loss: 0.6879 - accuracy: 0.5485 - val_loss: 0.6800 - val_accuracy: 0.5290
    Epoch 2/30
    100/100 [==============================] - 12s 124ms/step - loss: 0.6745 - accuracy: 0.5820 - val_loss: 0.6811 - val_accuracy: 0.5720
    Epoch 3/30
    100/100 [==============================] - 12s 124ms/step - loss: 0.6492 - accuracy: 0.6420 - val_loss: 0.6374 - val_accuracy: 0.6270
    Epoch 4/30
    100/100 [==============================] - 12s 123ms/step - loss: 0.6232 - accuracy: 0.6535 - val_loss: 0.6132 - val_accuracy: 0.6580
    Epoch 5/30
    100/100 [==============================] - 12s 123ms/step - loss: 0.6015 - accuracy: 0.6890 - val_loss: 0.6605 - val_accuracy: 0.6260
    Epoch 6/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.5710 - accuracy: 0.7015 - val_loss: 0.5549 - val_accuracy: 0.7080
    Epoch 7/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.5546 - accuracy: 0.7200 - val_loss: 0.5671 - val_accuracy: 0.6840
    Epoch 8/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.5548 - accuracy: 0.7130 - val_loss: 0.5724 - val_accuracy: 0.6970
    Epoch 9/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.5419 - accuracy: 0.7165 - val_loss: 0.5742 - val_accuracy: 0.7100
    Epoch 10/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.5299 - accuracy: 0.7405 - val_loss: 0.5731 - val_accuracy: 0.6880
    Epoch 11/30
    100/100 [==============================] - 12s 123ms/step - loss: 0.5143 - accuracy: 0.7425 - val_loss: 0.5387 - val_accuracy: 0.7230
    Epoch 12/30
    100/100 [==============================] - 12s 124ms/step - loss: 0.5128 - accuracy: 0.7330 - val_loss: 0.5297 - val_accuracy: 0.7300
    Epoch 13/30
    100/100 [==============================] - 12s 124ms/step - loss: 0.4898 - accuracy: 0.7600 - val_loss: 0.4999 - val_accuracy: 0.7620
    Epoch 14/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.5018 - accuracy: 0.7530 - val_loss: 0.5779 - val_accuracy: 0.7120
    Epoch 15/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4967 - accuracy: 0.7570 - val_loss: 0.4880 - val_accuracy: 0.7500
    Epoch 16/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4907 - accuracy: 0.7575 - val_loss: 0.5330 - val_accuracy: 0.7470
    Epoch 17/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4970 - accuracy: 0.7565 - val_loss: 0.5058 - val_accuracy: 0.7490
    Epoch 18/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4867 - accuracy: 0.7550 - val_loss: 0.4793 - val_accuracy: 0.7530
    Epoch 19/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4569 - accuracy: 0.7810 - val_loss: 0.4684 - val_accuracy: 0.7680
    Epoch 20/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4717 - accuracy: 0.7710 - val_loss: 0.4715 - val_accuracy: 0.7600
    Epoch 21/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4625 - accuracy: 0.7905 - val_loss: 0.5449 - val_accuracy: 0.7400
    Epoch 22/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4548 - accuracy: 0.7775 - val_loss: 0.4598 - val_accuracy: 0.7690
    Epoch 23/30
    100/100 [==============================] - 12s 123ms/step - loss: 0.4520 - accuracy: 0.7855 - val_loss: 0.4697 - val_accuracy: 0.7690
    Epoch 24/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4511 - accuracy: 0.7810 - val_loss: 0.4773 - val_accuracy: 0.7550
    Epoch 25/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4395 - accuracy: 0.7960 - val_loss: 0.4642 - val_accuracy: 0.7660
    Epoch 26/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4310 - accuracy: 0.8025 - val_loss: 0.4712 - val_accuracy: 0.7730
    Epoch 27/30
    100/100 [==============================] - 12s 123ms/step - loss: 0.4195 - accuracy: 0.8060 - val_loss: 0.4456 - val_accuracy: 0.7860
    Epoch 28/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4278 - accuracy: 0.7960 - val_loss: 0.4534 - val_accuracy: 0.7970
    Epoch 29/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4164 - accuracy: 0.8045 - val_loss: 0.4496 - val_accuracy: 0.7740
    Epoch 30/30
    100/100 [==============================] - 12s 122ms/step - loss: 0.4026 - accuracy: 0.8240 - val_loss: 0.4926 - val_accuracy: 0.7680
    Time : 0:06:10.407311
    


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


    
![png](/Machine-Learning/images/0420/output_15_0.png)
    


### epochs=100


```python
from timeit import default_timer as timer
from datetime import timedelta
start = timer()

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = './data/kaggle/cat_dog_small/train'  # cats folder, dogs folder
valid_dir = './data/kaggle/cat_dog_small/validation'

# ImageDataGenerator 생성
train_datagen = ImageDataGenerator(rescale=1/255,           # 정규화 포함
                                   rotation_range=30,       # 최대 회전 각도 (0~20)
                                   width_shift_range=0.1,   # 가로 방향으로 이미지 이동시키기 (0~10%)
                                   height_shift_range=0.1,  # 세로 방향으로 이미지 이동시키기 (0~50%) 
                                   zoom_range=0.2,          # 10%의 비율로 확대/축소
                                   horizontal_flip=True,    # 좌우 반전 (True:허용)
                                   fill_mode='nearest')     # 이미지 보정

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

model.add(Flatten())  # 3차원 => 1차원

model.add(Dense(units=256,
                activation='relu'))

model.add(Dense(units=1,
                activation='sigmoid'))

print(model.summary())

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,  # 2000개 이미지 20개씩 뽑아 100번
                    steps_per_epoch=100,
                    epochs=100,
                    validation_data=validation_generator,  # 1000개 이미지 20개씩 뽑아 50번
                    validation_steps=50)

model.save('./data/kaggle/cats_dogs_small_cnn_model_augmentation_epochs100.h5')

end = timer()
print(f'Time : {timedelta(seconds=end-start)}')
```

    Found 2000 images belonging to 2 classes.
    Found 1000 images belonging to 2 classes.
    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_8 (Conv2D)            (None, 148, 148, 32)      896       
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 74, 74, 32)        0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 72, 72, 64)        18496     
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 70, 70, 128)       73856     
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 35, 35, 128)       0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 33, 33, 128)       147584    
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 16, 16, 128)       0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 32768)             0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 256)               8388864   
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 8,629,953
    Trainable params: 8,629,953
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/100
    100/100 [==============================] - 13s 124ms/step - loss: 0.6887 - accuracy: 0.5355 - val_loss: 0.6832 - val_accuracy: 0.5920
    Epoch 2/100
    100/100 [==============================] - 12s 122ms/step - loss: 0.6507 - accuracy: 0.6100 - val_loss: 0.6364 - val_accuracy: 0.6140
    Epoch 3/100
    100/100 [==============================] - 14s 135ms/step - loss: 0.6244 - accuracy: 0.6560 - val_loss: 0.6201 - val_accuracy: 0.6560
    Epoch 4/100
    100/100 [==============================] - 13s 131ms/step - loss: 0.6039 - accuracy: 0.6770 - val_loss: 0.6151 - val_accuracy: 0.6440
    Epoch 5/100
    100/100 [==============================] - 13s 132ms/step - loss: 0.5959 - accuracy: 0.6775 - val_loss: 0.6364 - val_accuracy: 0.6440
    Epoch 6/100
    100/100 [==============================] - 13s 131ms/step - loss: 0.5811 - accuracy: 0.6935 - val_loss: 0.5550 - val_accuracy: 0.6970
    Epoch 7/100
    100/100 [==============================] - 13s 131ms/step - loss: 0.5783 - accuracy: 0.6925 - val_loss: 0.5553 - val_accuracy: 0.7120
    Epoch 8/100
    100/100 [==============================] - 13s 132ms/step - loss: 0.5473 - accuracy: 0.7180 - val_loss: 0.5241 - val_accuracy: 0.7260
    Epoch 9/100
    100/100 [==============================] - 13s 133ms/step - loss: 0.5380 - accuracy: 0.7275 - val_loss: 0.5470 - val_accuracy: 0.7020
    Epoch 10/100
    100/100 [==============================] - 13s 128ms/step - loss: 0.5390 - accuracy: 0.7320 - val_loss: 0.5297 - val_accuracy: 0.7230
    Epoch 11/100
    100/100 [==============================] - 13s 129ms/step - loss: 0.5233 - accuracy: 0.7405 - val_loss: 0.5081 - val_accuracy: 0.7430
    Epoch 12/100
    100/100 [==============================] - 13s 131ms/step - loss: 0.5124 - accuracy: 0.7500 - val_loss: 0.5047 - val_accuracy: 0.7490
    Epoch 13/100
    100/100 [==============================] - 13s 131ms/step - loss: 0.4985 - accuracy: 0.7445 - val_loss: 0.4914 - val_accuracy: 0.7510
    Epoch 14/100
    100/100 [==============================] - 13s 129ms/step - loss: 0.5046 - accuracy: 0.7640 - val_loss: 0.5049 - val_accuracy: 0.7440
    Epoch 15/100
    100/100 [==============================] - 13s 130ms/step - loss: 0.4937 - accuracy: 0.7580 - val_loss: 0.4930 - val_accuracy: 0.7580
    Epoch 16/100
    100/100 [==============================] - 13s 129ms/step - loss: 0.4876 - accuracy: 0.7650 - val_loss: 0.5322 - val_accuracy: 0.7330
    Epoch 17/100
    100/100 [==============================] - 13s 130ms/step - loss: 0.4896 - accuracy: 0.7620 - val_loss: 0.5101 - val_accuracy: 0.7510
    Epoch 18/100
    100/100 [==============================] - 13s 130ms/step - loss: 0.4708 - accuracy: 0.7680 - val_loss: 0.4955 - val_accuracy: 0.7530
    Epoch 19/100
    100/100 [==============================] - 13s 128ms/step - loss: 0.4718 - accuracy: 0.7725 - val_loss: 0.5011 - val_accuracy: 0.7430
    Epoch 20/100
    100/100 [==============================] - 13s 127ms/step - loss: 0.4603 - accuracy: 0.7735 - val_loss: 0.5090 - val_accuracy: 0.7340
    Epoch 21/100
    100/100 [==============================] - 13s 128ms/step - loss: 0.4604 - accuracy: 0.7890 - val_loss: 0.5158 - val_accuracy: 0.7320
    Epoch 22/100
    100/100 [==============================] - 13s 127ms/step - loss: 0.4643 - accuracy: 0.7740 - val_loss: 0.4594 - val_accuracy: 0.7730
    Epoch 23/100
    100/100 [==============================] - 13s 128ms/step - loss: 0.4559 - accuracy: 0.7890 - val_loss: 0.5207 - val_accuracy: 0.7510
    Epoch 24/100
    100/100 [==============================] - 13s 127ms/step - loss: 0.4370 - accuracy: 0.7890 - val_loss: 0.4811 - val_accuracy: 0.7600
    Epoch 25/100
    100/100 [==============================] - 13s 128ms/step - loss: 0.4442 - accuracy: 0.7965 - val_loss: 0.4622 - val_accuracy: 0.7730
    Epoch 26/100
    100/100 [==============================] - 13s 126ms/step - loss: 0.4360 - accuracy: 0.8000 - val_loss: 0.4869 - val_accuracy: 0.7700
    Epoch 27/100
    100/100 [==============================] - 13s 128ms/step - loss: 0.4344 - accuracy: 0.7930 - val_loss: 0.4787 - val_accuracy: 0.7640
    Epoch 28/100
    100/100 [==============================] - 13s 126ms/step - loss: 0.4259 - accuracy: 0.7975 - val_loss: 0.4504 - val_accuracy: 0.7780
    Epoch 29/100
    100/100 [==============================] - 13s 127ms/step - loss: 0.4150 - accuracy: 0.8110 - val_loss: 0.4564 - val_accuracy: 0.7770
    Epoch 30/100
    100/100 [==============================] - 13s 127ms/step - loss: 0.4188 - accuracy: 0.8085 - val_loss: 0.4657 - val_accuracy: 0.7820
    Epoch 31/100
    100/100 [==============================] - 13s 128ms/step - loss: 0.4103 - accuracy: 0.8150 - val_loss: 0.4903 - val_accuracy: 0.7610
    Epoch 32/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.4230 - accuracy: 0.8025 - val_loss: 0.4667 - val_accuracy: 0.7800
    Epoch 33/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.4129 - accuracy: 0.8155 - val_loss: 0.4599 - val_accuracy: 0.7870
    Epoch 34/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.4067 - accuracy: 0.8145 - val_loss: 0.4537 - val_accuracy: 0.7820
    Epoch 35/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.4042 - accuracy: 0.8130 - val_loss: 0.4738 - val_accuracy: 0.7760
    Epoch 36/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.4011 - accuracy: 0.8100 - val_loss: 0.4872 - val_accuracy: 0.7550
    Epoch 37/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.3889 - accuracy: 0.8150 - val_loss: 0.5033 - val_accuracy: 0.7690
    Epoch 38/100
    100/100 [==============================] - 12s 122ms/step - loss: 0.3882 - accuracy: 0.8225 - val_loss: 0.5738 - val_accuracy: 0.7210
    Epoch 39/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.3861 - accuracy: 0.8195 - val_loss: 0.4710 - val_accuracy: 0.7860
    Epoch 40/100
    100/100 [==============================] - 12s 122ms/step - loss: 0.3942 - accuracy: 0.8265 - val_loss: 0.4581 - val_accuracy: 0.7840
    Epoch 41/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.3647 - accuracy: 0.8510 - val_loss: 0.4613 - val_accuracy: 0.7820
    Epoch 42/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.3757 - accuracy: 0.8305 - val_loss: 0.4342 - val_accuracy: 0.7990
    Epoch 43/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.3684 - accuracy: 0.8230 - val_loss: 0.4446 - val_accuracy: 0.8000
    Epoch 44/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.3571 - accuracy: 0.8445 - val_loss: 0.4694 - val_accuracy: 0.7800
    Epoch 45/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.3583 - accuracy: 0.8430 - val_loss: 0.4680 - val_accuracy: 0.7870
    Epoch 46/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.3676 - accuracy: 0.8380 - val_loss: 0.4784 - val_accuracy: 0.7700
    Epoch 47/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.3553 - accuracy: 0.8375 - val_loss: 0.5075 - val_accuracy: 0.7700
    Epoch 48/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.3568 - accuracy: 0.8435 - val_loss: 0.4474 - val_accuracy: 0.7920
    Epoch 49/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.3543 - accuracy: 0.8450 - val_loss: 0.4725 - val_accuracy: 0.7940
    Epoch 50/100
    100/100 [==============================] - 13s 125ms/step - loss: 0.3538 - accuracy: 0.8445 - val_loss: 0.4417 - val_accuracy: 0.8020
    Epoch 51/100
    100/100 [==============================] - 13s 126ms/step - loss: 0.3379 - accuracy: 0.8475 - val_loss: 0.4724 - val_accuracy: 0.7970
    Epoch 52/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.3381 - accuracy: 0.8370 - val_loss: 0.4303 - val_accuracy: 0.7980
    Epoch 53/100
    100/100 [==============================] - 13s 125ms/step - loss: 0.3430 - accuracy: 0.8485 - val_loss: 0.4308 - val_accuracy: 0.8070
    Epoch 54/100
    100/100 [==============================] - 13s 126ms/step - loss: 0.3405 - accuracy: 0.8530 - val_loss: 0.5425 - val_accuracy: 0.7500
    Epoch 55/100
    100/100 [==============================] - 13s 127ms/step - loss: 0.3285 - accuracy: 0.8510 - val_loss: 0.4701 - val_accuracy: 0.7990
    Epoch 56/100
    100/100 [==============================] - 13s 125ms/step - loss: 0.3285 - accuracy: 0.8500 - val_loss: 0.4689 - val_accuracy: 0.7930
    Epoch 57/100
    100/100 [==============================] - 13s 126ms/step - loss: 0.3310 - accuracy: 0.8540 - val_loss: 0.4655 - val_accuracy: 0.7810
    Epoch 58/100
    100/100 [==============================] - 13s 126ms/step - loss: 0.3248 - accuracy: 0.8535 - val_loss: 0.4558 - val_accuracy: 0.8050
    Epoch 59/100
    100/100 [==============================] - 13s 126ms/step - loss: 0.3148 - accuracy: 0.8590 - val_loss: 0.4626 - val_accuracy: 0.7920
    Epoch 60/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.3089 - accuracy: 0.8640 - val_loss: 0.5083 - val_accuracy: 0.7860
    Epoch 61/100
    100/100 [==============================] - 13s 125ms/step - loss: 0.3085 - accuracy: 0.8635 - val_loss: 0.4725 - val_accuracy: 0.8000
    Epoch 62/100
    100/100 [==============================] - 13s 126ms/step - loss: 0.3028 - accuracy: 0.8630 - val_loss: 0.5098 - val_accuracy: 0.7740
    Epoch 63/100
    100/100 [==============================] - 13s 128ms/step - loss: 0.3147 - accuracy: 0.8630 - val_loss: 0.5357 - val_accuracy: 0.7790
    Epoch 64/100
    100/100 [==============================] - 13s 128ms/step - loss: 0.2922 - accuracy: 0.8755 - val_loss: 0.4800 - val_accuracy: 0.7950
    Epoch 65/100
    100/100 [==============================] - 13s 126ms/step - loss: 0.2913 - accuracy: 0.8775 - val_loss: 0.5222 - val_accuracy: 0.7770
    Epoch 66/100
    100/100 [==============================] - 12s 125ms/step - loss: 0.2793 - accuracy: 0.8765 - val_loss: 0.4872 - val_accuracy: 0.7850
    Epoch 67/100
    100/100 [==============================] - 12s 125ms/step - loss: 0.2875 - accuracy: 0.8715 - val_loss: 0.4863 - val_accuracy: 0.8000
    Epoch 68/100
    100/100 [==============================] - 13s 125ms/step - loss: 0.2878 - accuracy: 0.8705 - val_loss: 0.4493 - val_accuracy: 0.8160
    Epoch 69/100
    100/100 [==============================] - 13s 125ms/step - loss: 0.2730 - accuracy: 0.8830 - val_loss: 0.5042 - val_accuracy: 0.7990
    Epoch 70/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2816 - accuracy: 0.8745 - val_loss: 0.6109 - val_accuracy: 0.7540
    Epoch 71/100
    100/100 [==============================] - 13s 125ms/step - loss: 0.2722 - accuracy: 0.8815 - val_loss: 0.4706 - val_accuracy: 0.8210
    Epoch 72/100
    100/100 [==============================] - 13s 125ms/step - loss: 0.2769 - accuracy: 0.8775 - val_loss: 0.4664 - val_accuracy: 0.8220
    Epoch 73/100
    100/100 [==============================] - 13s 126ms/step - loss: 0.2788 - accuracy: 0.8810 - val_loss: 0.4560 - val_accuracy: 0.8240
    Epoch 74/100
    100/100 [==============================] - 13s 126ms/step - loss: 0.2729 - accuracy: 0.8810 - val_loss: 0.5356 - val_accuracy: 0.7860
    Epoch 75/100
    100/100 [==============================] - 12s 125ms/step - loss: 0.2846 - accuracy: 0.8680 - val_loss: 0.4932 - val_accuracy: 0.8050
    Epoch 76/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2772 - accuracy: 0.8785 - val_loss: 0.4957 - val_accuracy: 0.7940
    Epoch 77/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.2593 - accuracy: 0.8895 - val_loss: 0.5792 - val_accuracy: 0.7790
    Epoch 78/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2668 - accuracy: 0.8800 - val_loss: 0.5029 - val_accuracy: 0.8030
    Epoch 79/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.2792 - accuracy: 0.8790 - val_loss: 0.4901 - val_accuracy: 0.8130
    Epoch 80/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2612 - accuracy: 0.8860 - val_loss: 0.5262 - val_accuracy: 0.8020
    Epoch 81/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.2416 - accuracy: 0.9015 - val_loss: 0.5438 - val_accuracy: 0.7870
    Epoch 82/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2576 - accuracy: 0.8940 - val_loss: 0.5083 - val_accuracy: 0.8090
    Epoch 83/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.2285 - accuracy: 0.9085 - val_loss: 0.5316 - val_accuracy: 0.7970
    Epoch 84/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2429 - accuracy: 0.8900 - val_loss: 0.5733 - val_accuracy: 0.7940
    Epoch 85/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.2514 - accuracy: 0.8940 - val_loss: 0.5143 - val_accuracy: 0.8050
    Epoch 86/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2583 - accuracy: 0.8860 - val_loss: 0.5080 - val_accuracy: 0.8000
    Epoch 87/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.2386 - accuracy: 0.9010 - val_loss: 0.5289 - val_accuracy: 0.7950
    Epoch 88/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2385 - accuracy: 0.9075 - val_loss: 0.5035 - val_accuracy: 0.8150
    Epoch 89/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.2341 - accuracy: 0.9060 - val_loss: 0.5023 - val_accuracy: 0.8070
    Epoch 90/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2318 - accuracy: 0.9030 - val_loss: 0.5391 - val_accuracy: 0.7960
    Epoch 91/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.2482 - accuracy: 0.8970 - val_loss: 0.5373 - val_accuracy: 0.7860
    Epoch 92/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2134 - accuracy: 0.9145 - val_loss: 0.5352 - val_accuracy: 0.8160
    Epoch 93/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.2450 - accuracy: 0.9000 - val_loss: 0.5091 - val_accuracy: 0.8110
    Epoch 94/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2130 - accuracy: 0.9130 - val_loss: 0.5330 - val_accuracy: 0.8050
    Epoch 95/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.2017 - accuracy: 0.9215 - val_loss: 0.5666 - val_accuracy: 0.8040
    Epoch 96/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2319 - accuracy: 0.8975 - val_loss: 0.5031 - val_accuracy: 0.8140
    Epoch 97/100
    100/100 [==============================] - 12s 124ms/step - loss: 0.1883 - accuracy: 0.9265 - val_loss: 0.5333 - val_accuracy: 0.8180
    Epoch 98/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2302 - accuracy: 0.9070 - val_loss: 0.5697 - val_accuracy: 0.7990
    Epoch 99/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.1995 - accuracy: 0.9190 - val_loss: 0.5842 - val_accuracy: 0.7950
    Epoch 100/100
    100/100 [==============================] - 12s 123ms/step - loss: 0.2310 - accuracy: 0.9050 - val_loss: 0.5374 - val_accuracy: 0.7950
    Time : 0:21:01.547628
    


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


    
![png](/Machine-Learning/images/0420/output_18_0.png)
    


## Transfer Learning
- keras는 전이학습을 위해 VGG16 제공


```python
from tensorflow.keras.applications import VGG16

model_base = VGG16(weights='imagenet',  # 학습할 이미지 dataset
                   include_top=True,    # classification 역할의 FC layer 포함 여부
                                        # True: FC layer 포함
                   input_shape=(224, 224, 3))

print(model_base.summary())
```

    Model: "vgg16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    fc1 (Dense)                  (None, 4096)              102764544 
    _________________________________________________________________
    fc2 (Dense)                  (None, 4096)              16781312  
    _________________________________________________________________
    predictions (Dense)          (None, 1000)              4097000   
    =================================================================
    Total params: 138,357,544
    Trainable params: 138,357,544
    Non-trainable params: 0
    _________________________________________________________________
    None
    


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
    

- 내가 갖고 있는 데이터(개와 고양이)를 VGG16에 통과시켜서 activation map 만들기
- ndarray 형태로 저장


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
    Time : 0:00:24.925941
    

- 만든 activation map을 이용해서 DNN 학습


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
    100/100 [==============================] - 1s 4ms/step - loss: 0.4593 - accuracy: 0.7710 - val_loss: 0.3172 - val_accuracy: 0.8680
    Epoch 2/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.2827 - accuracy: 0.8805 - val_loss: 0.2736 - val_accuracy: 0.8870
    Epoch 3/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.2238 - accuracy: 0.9135 - val_loss: 0.2582 - val_accuracy: 0.8970
    Epoch 4/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.1880 - accuracy: 0.9280 - val_loss: 0.2408 - val_accuracy: 0.8990
    Epoch 5/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.1739 - accuracy: 0.9335 - val_loss: 0.2458 - val_accuracy: 0.8960
    Epoch 6/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.1406 - accuracy: 0.9510 - val_loss: 0.2386 - val_accuracy: 0.9010
    Epoch 7/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.1232 - accuracy: 0.9605 - val_loss: 0.2322 - val_accuracy: 0.9060
    Epoch 8/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.1014 - accuracy: 0.9700 - val_loss: 0.2384 - val_accuracy: 0.9030
    Epoch 9/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0935 - accuracy: 0.9770 - val_loss: 0.2354 - val_accuracy: 0.8990
    Epoch 10/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0853 - accuracy: 0.9725 - val_loss: 0.2411 - val_accuracy: 0.8990
    Epoch 11/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0708 - accuracy: 0.9800 - val_loss: 0.2702 - val_accuracy: 0.8930
    Epoch 12/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0581 - accuracy: 0.9860 - val_loss: 0.2549 - val_accuracy: 0.8990
    Epoch 13/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0516 - accuracy: 0.9900 - val_loss: 0.2412 - val_accuracy: 0.9030
    Epoch 14/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0481 - accuracy: 0.9925 - val_loss: 0.2464 - val_accuracy: 0.9050
    Epoch 15/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0410 - accuracy: 0.9925 - val_loss: 0.2637 - val_accuracy: 0.9040
    Epoch 16/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0363 - accuracy: 0.9960 - val_loss: 0.2580 - val_accuracy: 0.8990
    Epoch 17/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0325 - accuracy: 0.9975 - val_loss: 0.2656 - val_accuracy: 0.9010
    Epoch 18/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0260 - accuracy: 0.9985 - val_loss: 0.2845 - val_accuracy: 0.9000
    Epoch 19/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0258 - accuracy: 0.9990 - val_loss: 0.2707 - val_accuracy: 0.9070
    Epoch 20/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0222 - accuracy: 0.9980 - val_loss: 0.2802 - val_accuracy: 0.9050
    Epoch 21/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0205 - accuracy: 1.0000 - val_loss: 0.2790 - val_accuracy: 0.9050
    Epoch 22/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0196 - accuracy: 0.9995 - val_loss: 0.3150 - val_accuracy: 0.9000
    Epoch 23/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0165 - accuracy: 0.9990 - val_loss: 0.3031 - val_accuracy: 0.9040
    Epoch 24/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0138 - accuracy: 1.0000 - val_loss: 0.3123 - val_accuracy: 0.9000
    Epoch 25/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0130 - accuracy: 1.0000 - val_loss: 0.2949 - val_accuracy: 0.9060
    Epoch 26/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0120 - accuracy: 1.0000 - val_loss: 0.3065 - val_accuracy: 0.9040
    Epoch 27/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0103 - accuracy: 1.0000 - val_loss: 0.3013 - val_accuracy: 0.9080
    Epoch 28/30
    100/100 [==============================] - 0s 5ms/step - loss: 0.0107 - accuracy: 0.9995 - val_loss: 0.3116 - val_accuracy: 0.9070
    Epoch 29/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0088 - accuracy: 1.0000 - val_loss: 0.3166 - val_accuracy: 0.9080
    Epoch 30/30
    100/100 [==============================] - 0s 4ms/step - loss: 0.0096 - accuracy: 1.0000 - val_loss: 0.3343 - val_accuracy: 0.8960
    Time : 0:00:12.965057
    
