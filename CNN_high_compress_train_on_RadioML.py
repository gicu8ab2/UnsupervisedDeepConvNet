import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LeakyReLU, PReLU
from keras.regularizers import *
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random, sys, keras
import h5py
K.set_image_dim_ordering('th')

# Set params
epochs = 100     # number of epochs to train on
batch_size = 1024  # training batch size
nb_classes = 11
nb_channels = 1
nb_rows = 2
nb_cols = 128

### Load and prep the dataset #######
f = h5py.File('./data/RadioML_training_data.h5','r')
X_train = f['training_data'][()]
f = h5py.File('./data/RadioML_training_labels.h5','r')
Y_train = f['training_labels'][()]
f = h5py.File('./data/RadioML_testing_data.h5','r')
X_test = f['testing_data'][()]
f = h5py.File('./data/RadioML_testing_labels.h5','r')
Y_test = f['testing_labels'][()]
####################################################


### CNN high compression model #############################
in_shp = list(X_test.shape[1:])
dr = 0.5 # dropout rate (%)
model = models.Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(256, kernel_size=(1,3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(80, kernel_size=(2,3), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', name="dense2"))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal', name="dense3"))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal', name="dense4"))
model.add(Dense( nb_classes, kernel_initializer='he_normal', name="dense5" ))
model.add(Activation('softmax'))
model.add(Reshape([nb_classes]))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
model.summary()
filepath = 'CNN_high_compression.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)
###################################################




