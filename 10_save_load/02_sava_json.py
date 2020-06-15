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
for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)
# for i in train_samples:
#     print(i)   
# for i in train_labels:
#     print(i)
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
print('*********************')
print ('train_samples:', train_samples.shape)
print ('train_labels:', train_labels.shape)
print('*********************')
train_labels, train_samples = shuffle(train_labels, train_samples)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
# for i in scaled_train_samples:
#     print(i)
# print('\nscaled_train_samples[:5]:')
# print(scaled_train_samples[:5])
# print('train_labels[:5]:', train_labels[:5])
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

# model fit
print('*********************')
print ('scaled_train_samples:', scaled_train_samples.shape)
print ('train_labels:', train_labels.shape)
print('*********************')
model.fit(x=scaled_train_samples, y=train_labels, \
    batch_size=10, epochs=2, verbose=2)
# 1. model.save()

# This save functions saves:
# The architecture of the model, allowing to re-create the model.
# The weights of the model.
# The training configuration (loss, optimizer).
# The state of the optimizer, allowing to resume training exactly 
# where you left off.

# Checks first to see if file exists already.
# If not, the model is saved to disk.
import os.path

if os.path.isfile('models/medical_trial_model.h5') is False:
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    model.save('models/medical_trial_model.h5')

if os.path.isfile('models/medical_trial_model.h5') is False:
    model.save('models/medical_trial_model.h5')

# 2. model.to_json()

# save as JSON
json_string = model.to_json()

# save as YAML
# yaml_string = model.to_yaml()

# model reconstruction from JSON:
from tensorflow.keras.models import model_from_json
model_architecture = model_from_json(json_string)

# model reconstruction from YAML
# from tensorflow.keras.models import model_from_yaml
# model = model_from_yaml(yaml_string)

model_architecture.summary()
