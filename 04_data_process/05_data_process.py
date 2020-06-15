import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

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

# Data Processing
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)
print('************')
test = np.array ([1, 2, 3, 4, 5])
print('test.shape:')
print(test.shape)
test1 = np.array ([[1, 2, 3, 4, 5]])
print('test1.shape:')
print(test1.shape)
test2 = np.array ([[1], [2], [3], [4], [5]])
print('test2.shape:')
print(test2.shape)
print('train_samples.shape:')
print(train_samples.shape)
print('train_labels.shape:')
print(train_labels.shape)
print('************')
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
print('************')
print('scaled_train_samples.shape:')
print(scaled_train_samples.shape)
print('train_labels.shape:')
print(train_labels.shape)
print('************')
# for i in scaled_train_samples:
#     print(i)