import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend as K
from keras.models import Model
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import random, sys, keras
import pickle
K.set_image_dim_ordering('th')
import h5py

# Set params
nb_classes = 11
nb_channels = 1
nb_rows = 2
nb_cols = 128

#Load data
f = h5py.File('./data/RadioML_testing_data.h5','r')
X_test = f['testing_data'][()]


#Build model structure
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

#load model weights
filepath = './saved_models/CNN_high_compression.h5'
model.load_weights(filepath)

#Extract features
model_features = Model(inputs=model.input,outputs=model.get_layer('dense4').output)
test_feature_vecs=model_features.predict(X_test)

f=h5py.File('feature_vectors/test_features_cnn_high_compression.h5','w')
f.create_dataset('test_feature_vecs',data=test_feature_vecs)
f.close()





