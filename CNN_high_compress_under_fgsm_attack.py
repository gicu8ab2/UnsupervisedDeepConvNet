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
f = h5py.File('./data/X_adv_from_vtcnn2.h5','r')
X_adv = f['X_adv'][()]
X_adv = np.std(X_test.flatten())*(X_adv-np.mean(X_adv.flatten()))/np.std(X_adv.flatten())

with open("./data/RML2016.10a_dict.dat", 'rb') as f:
    Xd = pickle.load(f, encoding='latin1')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
in_shp = list(X_test.shape[1:])

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

Y_test_hat=model.predict(X_adv)
y_test_hat=np.argmax(Y_test_hat,axis=1)

#Compute performance across all SNRs
f = h5py.File('./data/RadioML_testing_labels.h5','r')
Y_test = f['testing_labels'][()]
y_test=np.argmax(Y_test,axis=1)

print(sum(y_test_hat==y_test)/len(y_test))

# Create confusion matrix on per SNR basis
np.random.seed(2016)
n_train = 110000
train_idx = np.random.choice(range(0,220000), size=n_train, replace=False)
test_idx = list(set(range(0,220000))-set(train_idx))

acc = {}
acc_adv = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_X_adv_i = X_adv[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    test_Y_i_adv_hat = model.predict(test_X_adv_i)
    conf = np.zeros([nb_classes,nb_classes])
    confnorm = np.zeros([nb_classes,nb_classes])
    conf_adv = np.zeros([nb_classes,nb_classes])
    confnorm_adv = np.zeros([nb_classes,nb_classes])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
        k_adv = int(np.argmax(test_Y_i_adv_hat[i,:]))
        conf_adv[j,k_adv] = conf_adv[j,k_adv] + 1
    for i in range(0,nb_classes):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        confnorm_adv[i,:] = conf_adv[i,:] / np.sum(conf_adv[i,:])
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)

    cor_adv = np.sum(np.diag(conf_adv))
    ncor_adv = np.sum(conf_adv) - cor_adv
    print("Overall Accuracy: ", cor_adv / (cor_adv+ncor_adv))
    acc_adv[snr] = 1.0*cor_adv/(cor_adv+ncor_adv)


##### save results ##################
fd = open('./results/accuracy_CNN_high_compression.dat','wb')
pickle.dump(acc, fd )
fd.close()

fd = open('./results/accuracy_CNN_high_compression_adv.dat','wb')
pickle.dump(acc_adv, fd )
fd.close()
