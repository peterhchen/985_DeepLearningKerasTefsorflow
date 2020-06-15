import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
# data preprocessed
train_labels = []
train_samples = []
valid_labels = []
valid_samples = []
for i in range(5):
    # The ~0.5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    valid_samples.append(random_younger)
    valid_labels.append(1)

    # The ~0.5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    valid_samples.append(random_older)
    valid_labels.append(0)

for i in range(45):
    # The ~4.5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~4.5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(100):
    # The ~5% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    valid_samples.append(random_younger)
    valid_labels.append(0)

    # The ~5% of older individuals who did experience side effects
    random_older = randint(65,100)
    valid_samples.append(random_older)
    valid_labels.append(1)

for i in range(900):
    # The ~86.5% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~86.5% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)
# for i in train_samples:
#     print(i)   
# for i in train_labels:
#     print(i)
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

valid_labels = np.array(valid_labels)
valid_samples = np.array(valid_samples)
valid_labels, valid_samples = shuffle(valid_labels, valid_samples)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_valid_samples = scaler.fit_transform(valid_samples.reshape(-1,1))
# for i in scaled_train_samples:
#     print(i)
print('\nscaled_train_samples[:5]:')
print(scaled_train_samples[:5])
print('train_labels[:5]:', train_labels[:5])

# Set up Cuda GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if (len(physical_devices) > 0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Set up modle
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])
# model summary
model.summary()

# model compile
model.compile(optimizer=Adam(learning_rate=0.0001), \
    loss='sparse_categorical_crossentropy', metrics=['accuracy'])

valid_set = (scaled_valid_samples, valid_labels)
# model fit
model.fit(x=scaled_train_samples, y=train_labels, validation_data=valid_set, \
    batch_size=10, epochs=30, verbose=2)