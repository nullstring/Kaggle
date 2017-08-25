import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from keras.models import Sequential
from keras.layers import Input,Dropout,Flatten,Conv2D,MaxPooling2D,Dense,Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras import backend as K

K.set_image_data_format('channels_first')

DATASET_DIR = '/Users/harshm/Documents/GitHub/Kaggle/Datasets/dogs-vs-cats-redux-kernels-edition/'
TRAIN_DIR = DATASET_DIR + 'train/'
TEST_DIR = DATASET_DIR + 'test/'

ROWS = 128
COLS = 128
CHANNELS = 3

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
train_dogs = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]


test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) # cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%250 == 0: print ('Processed {} of {}'.format(i, count))
    
    return data

train = prep_data(train_images)
test = prep_data(test_images)

print ('Train shape: {}'.format(train.shape))
print ('Test shape: {}'.format(test.shape))

# MARK: generate labels

labels = []
for i in train_images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)
#sns.countplot(labels)
#sns.plt.title('Cats and Dogs')

# MARK: scaled down version of VGG-16
# 
# 1. Num conv cut in half, dense layers scaled down
# 2. RMSprop to optimize
# 3. Sigmoid for binary cross-entropy instead of softmax
# 4. Some other layers commented out

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

def catdog():
    model  = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, ROWS, COLS), activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    return model

model = catdog()

# MARK: Training

nb_epoch = 10
batch_size = 16

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_losses'))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

def run_catdog():
    history = LossHistory()
    model.fit(train, labels, batch_size=batch_size, epochs=nb_epoch,
            validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])
    predictions = model.predict(test, verbose=0)
    return predictions, history

predictions, history = run_catdog()
loss = history_losses
val_loss = history.val_losses

print(loss)
print (val_loss)
