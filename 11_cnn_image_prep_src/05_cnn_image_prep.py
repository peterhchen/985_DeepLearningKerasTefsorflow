import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt

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
for i in random.sample(glob.glob('train/cat*'), 500):
    #print ('i:', i)
    shutil.copy(i, 'train/cat')      
for i in random.sample(glob.glob('train/dog*'), 500):
    shutil.copy(i, 'train/dog')
for i in random.sample(glob.glob('train/cat*'), 100):
    shutil.copy(i, 'valid/cat')        
for i in random.sample(glob.glob('train/dog*'), 100):
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
for i in random.sample(glob.glob('test1/*'), 50):
    shutil.copy(i, 'test/cat')      
for i in random.sample(glob.glob('test1/*'), 50):
    shutil.copy(i, 'test/dog')
os.chdir('../../')

# process the data
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if (len(physical_devices) > 0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_path = 'data/dogs-vs-cats/train'
valid_path = 'data/dogs-vs-cats/valid'
test_path = 'data/dogs-vs-cats/test'

train_batches = ImageDataGenerator(\
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), \
    classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(\
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), \
        classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(\
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), \
        classes=['cat', 'dog'], batch_size=10, shuffle=False)

imgs, labels = next(train_batches)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(imgs)
print(labels)