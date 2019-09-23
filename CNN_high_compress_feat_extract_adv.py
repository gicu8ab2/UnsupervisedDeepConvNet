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
f = h5py.File('./data/RadioML_training_data.h5','r')
X_train = f['training_data'][()]
f = h5py.File('./data/X_adv_from_vtcnn2.h5','r')
X_adv = f['X_adv'][()]
#tmp = np.load('./data/X_adv_FGSM_from_CNN2.npz')
#X_adv = tmp['X_adv']
X_adv = np.std(X_train.flatten())*(X_adv-np.mean(X_adv.flatten()))/np.std(X_adv.flatten())

in_shp = list(X_train.shape[1:])

#Build model structure
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
adv_feature_vecs=model_features.predict(X_adv)

f=h5py.File('feature_vectors/adv_features_cnn_high_compression.h5','w')
f.create_dataset('adv_feature_vecs',data=adv_feature_vecs)
f.close()





