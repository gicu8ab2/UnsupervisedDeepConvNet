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
import pickle
from attacks.fgsm import fgsm
K.set_image_dim_ordering('th')
from resnet_2D_classify import ResnetBuilder

# Set params
nb_classes = 11
nb_channels = 1
nb_rows = 2
nb_cols = 128
epochs = 50     # number of epochs to train on
batch_size = 64  # training batch size

# Load the dataset ...
with open("../data/RML2016.10a_dict.dat", 'rb') as f:
    Xd = pickle.load(f, encoding='latin1')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
# Partition the data into training and test sets
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.5)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
in_shp = list(X_train.shape[1:])
X_train.shape, in_shp
classes = mods
sess = tf.InteractiveSession()
K.set_session(sess)

#Reshape inputs to comply with Conv2D standard of 4-D tensor (theano ordering)
X_test=np.reshape(X_test,[X_test.shape[0],1,X_test.shape[1],X_test.shape[2]])

### Rob's ResNet model #############################
model = ResnetBuilder.build_resnet_34((nb_channels,nb_rows,nb_cols), nb_classes)
model.compile(loss="categorical_crossentropy", optimizer="sgd")

filepath = '../saved_models/resnet34_wts_50_epochs.h5'
if False:
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=epochs,
                        verbose=2, validation_data=(X_test, Y_test),
                        callbacks = [
                        ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                        save_best_only=True, mode='auto'),
                        EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                        ])
    # we re-load the best weights once training is finished
    model.load_weights(filepath)
else:
    model.load_weights(filepath)

#load adversarial test samples generated from VTCNN2
tmp = np.load('../data/radio_ml_adversarial.npz')
X_adv = tmp['X_adv']
X_adv = np.std(X_train.flatten())*(X_adv-np.mean(X_adv.flatten()))/np.std(X_adv.flatten())

#Reshape inputs to comply with Conv2D standard of 4-D tensor (theano ordering)
X_adv=np.reshape(X_adv,[X_adv.shape[0],1,X_adv.shape[1],X_adv.shape[2]])

# Create confusion matrix on per SNR basis
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
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    conf_adv = np.zeros([len(classes),len(classes)])
    confnorm_adv = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
        k_adv = int(np.argmax(test_Y_i_adv_hat[i,:]))
        conf_adv[j,k_adv] = conf_adv[j,k_adv] + 1
    for i in range(0,len(classes)):
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

#save accuracy results
fd = open('../results/results_resnet34.dat','wb')
pickle.dump(acc, fd )
fd.close()
fd = open('../results/results_resnet34_adversarial_black.dat','wb')
pickle.dump(acc_adv, fd )
fd.close()



