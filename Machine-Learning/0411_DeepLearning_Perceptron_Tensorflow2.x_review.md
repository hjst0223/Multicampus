# tensorflow 2.x 버전으로 구현하기 😣


```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report

# Training Data Set
x_data = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float64)

# XOR GATE 연산에 대한 t_data
t_data = np.array([0, 1, 1, 0], dtype=np.float64)

# Tensorflow 구현
model = Sequential()

model.add(Flatten(input_shape=(2,)))   # input layer
model.add(Dense(units=128,
                activation='sigmoid'))  # 1번째 hidden layer
model.add(Dense(units=32,
                activation='sigmoid'))  # 2번째 hidden layer
model.add(Dense(units=16,
                activation='sigmoid'))  # 3번째 hidden layer
model.add(Dense(units=1,
                activation='sigmoid'))  # output layer

model.compile(optimizer=SGD(learning_rate=1e-2),
              loss='binary_crossentropy')

model.fit(x_data,
          t_data.reshape(-1, 1),
          epochs=30000,
          verbose=0)
```




    <tensorflow.python.keras.callbacks.History at 0x1a4d0bb4c40>



## evaluation (모델 평가)


```python
predict_val = model.predict(x_data)
predict_val = (tf.cast(predict_val > 0.5, dtype=tf.float32)).numpy().ravel()

print(classification_report(t_data, predict_val))
```

                  precision    recall  f1-score   support
    
             0.0       0.50      0.50      0.50         2
             1.0       0.50      0.50      0.50         2
    
        accuracy                           0.50         4
       macro avg       0.50      0.50      0.50         4
    weighted avg       0.50      0.50      0.50         4
    
    
