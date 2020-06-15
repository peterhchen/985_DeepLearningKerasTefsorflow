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
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import confusion_matrix
import itertools

# plots images with labels within jupyter notebook
# source: https://github.com/fastai/courses/blob/master/deeplearning1/nbs/utils.py#L79
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.show()
 
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
    validation_data=valid_batches, validation_steps=4, \
    epochs=10, verbose=2)

test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles=test_labels)
print ('\ntest_labels:')
print(test_labels)
# test_labels = test_labels[:,0]
# print ('\ntest_labels1:')
# print(test_labels)

predictions = model.predict(x=test_batches, steps=1, verbose=0)
print('\npredictions:')
print(predictions)
# cm = confusion_matrix(y_true=test_labels, \
#     y_pred=np.round(predictions[:,0]))
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

cm_plot_labels = ['cat','dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, \
    title='Confusion Matrix')

if os.path.isfile('models/VGG16_cats_and_dogs.h5') is False:
    model.save('models/VGG16_cats_and_dogs.h5')