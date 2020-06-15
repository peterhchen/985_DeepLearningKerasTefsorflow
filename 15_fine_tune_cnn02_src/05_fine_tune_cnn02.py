import tensorflow as tf
import numpy as np
import os
import shutil
import random
import glob
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Organize data into train, valid, test dirs
os.chdir('data/dogs-vs-cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    if os.path.isdir('valid/dog') is False:
        os.makedirs('valid/dog')
    if os.path.isdir('valid/cat') is False:
        os.makedirs('valid/cat')
    if os.path.isdir('test/dog') is False:
        os.makedirs('test/dog')
    if os.path.isdir('test/cat') is False:
        os.makedirs('test/cat')

# copy the data the target
# print ('*******************************************')
# print ("glob.glob('train/cat*'):",glob.glob('train/cat*'))
# print ("random.sample(glob.glob('train/cat*'), 500):",random.sample(glob.glob('train/cat*'), 500))
# print ('*******************************************')

for file in os.scandir('train/dog'):
    os.unlink(file)
for file in os.scandir('train/cat'):
    os.unlink(file)

for i in random.sample(glob.glob('train/cat*'), 50):
        #print ('i:', i)
    shutil.copy(i, 'train/cat')      
for i in random.sample(glob.glob('train/dog*'), 50):
    shutil.copy(i, 'train/dog')
for i in random.sample(glob.glob('train/cat*'), 10):
    shutil.copy(i, 'valid/cat')        
for i in random.sample(glob.glob('train/dog*'), 10):
    # print ('line 59 i:', i)
    shutil.copy(i, 'valid/dog')
# print ('*******************************************')
# print ("glob.glob('test1/*'):",glob.glob('test1/*'))
# print ("random.sample(glob.glob('test1/*'), 500):",random.sample(glob.glob('test1/*'), 500))
# print ('*******************************************')
for file in os.scandir('test/dog'):
    os.unlink(file)
for file in os.scandir('test/cat'):
    os.unlink(file)

# for i in random.sample(glob.glob('test1/*'), 50):
#     shutil.copy(i, 'test/cat')      
# for i in random.sample(glob.glob('test1/*'), 50):
#     shutil.copy(i, 'test/dog')
for i in random.sample(glob.glob('test1/*'), 5):
    shutil.copy(i, 'test/cat')      
for i in random.sample(glob.glob('test1/*'), 5):
    shutil.copy(i, 'test/dog')
os.chdir('../../')

train_path = 'data/dogs-vs-cats/train'
valid_path = 'data/dogs-vs-cats/valid'

train_batches = ImageDataGenerator(\
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), \
    classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(\
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), \
        classes=['cat', 'dog'], batch_size=10)

# Download model - Internet connection needed
vgg16_model = tf.keras.applications.vgg16.VGG16()
print('*********************************')
print ('vgg16_model:', vgg16_model)
print('*********************************')
# print the vgg16_model
vgg16_model.summary()

def count_params(model):
    non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    return {'non_trainable_params': non_trainable_params, 'trainable_params': trainable_params}

# Check the vgg16_model summary, we see the trainable parameters is 138357544
params = count_params(vgg16_model)
assert params['non_trainable_params'] == 0
assert params['trainable_params'] == 138357544

# print type of model
print('type(vgg16_model): ', type(vgg16_model))

# Create empty Sequentail model
model = Sequential()
# Add layer by layer from the vgg16_model to our Sequential model
# Skip the last layer (index = -1) pf vgg16_model.
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
# print our new Sequential Model (witout the last layer)
model.summary()

# Check the model summary, we see the trainable parameters is 134260544
params = count_params(model)
assert params['non_trainable_params'] == 0
assert params['trainable_params'] == 134260544

# Set the layer to non-trainable
for layer in model.layers:
    layer.trainable = False

# Add the Dense layer to the last output layer
model.add(Dense(units=2, activation='softmax'))
model.summary()

# Compiled the fine-tuned vgg16_model
model.compile(optimizer=Adam(learning_rate=0.0001), \
    loss='categorical_crossentropy', metrics=['accuracy'])
# model fit
model.fit(x=train_batches, steps_per_epoch=4, \
    validation_data=valid_batches, validation_steps=4, epochs=10, verbose=2)