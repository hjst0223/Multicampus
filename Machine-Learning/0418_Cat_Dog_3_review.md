# 데이터 나누기 🐱‍👤
### 각각 12,500개의 (고양이와 개의) 이미지 파일을 다음과 같이 분배
- train : 7,000 
- validation : 3,000
- test : 2,500 


```python
import os, shutil

original_dataset_dir = './data/kaggle/cat_dog/train'

base_dir = 'data/kaggle/cat_dog_full'
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


fnames = ['cat.{}.jpg'.format(i) for i in range(7000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(train_cats_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(7000,10000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(validation_cats_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(10000,12500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(test_cats_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(7000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(train_dogs_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(7000,10000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(validation_dogs_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(10000,12500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname).replace('\\','/')
    dst = os.path.join(test_dogs_dir, fname).replace('\\','/')
    shutil.copyfile(src,dst)
```

# Keras가 제공하는 ImageDataGenerator 사용하기 😗


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
    


    
![png](/Machine-Learning/images/0418/output_3_1.png)
    

