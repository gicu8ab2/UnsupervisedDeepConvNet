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
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random, sys, keras
import pickle
import h5py
from attacks.fgsm import fgsm
K.set_image_dim_ordering('th')

# Set params
nb_classes = 11
img_rows = 2
img_cols = 128
epochs = 100     # number of epochs to train on
batch_size = 1024  # training batch size
batch_size_adv = 64  # batch size for adversarial 

# Load the dataset ...
with open("RML2016.10a_dict.dat", 'rb') as f:
    Xd = pickle.load(f, encoding='latin1')

snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data
#  into training and test sets of the form we can train/test on 
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

# f=h5py.File('RadioML_training_data.h5','w')
# f.create_dataset('training_data',data=X_train)
# f.close()

# f=h5py.File('RadioML_training_labels.h5','w')
# f.create_dataset('training_labels',data=Y_train)
# f.close()

# f=h5py.File('RadioML_testing_data.h5','w')
# f.create_dataset('testing_data',data=X_test)
# f.close()

# f=h5py.File('RadioML_testing_labels.h5','w')
# f.create_dataset('testing_labels',data=Y_test)
# f.close()


# Build VT-CNN2 model for use with adversarial generation
dr = .5
model = models.Sequential([
    Reshape([1]+in_shp, input_shape=in_shp),
    ZeroPadding2D((0, 2)),
    Conv2D(256, kernel_size=(1,3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform'),
    Dropout(dr),
    ZeroPadding2D((0, 2)),
    Conv2D(80, kernel_size=(2,3), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'),
    Dropout(dr),
    Flatten(),
    Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"),
    Dropout(dr),
    Dense( len(classes), kernel_initializer='he_normal', name="dense2" ),
    Activation('softmax'),
    Reshape([len(classes)])
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
#filepath = 'saved_models/VTCNN2_wts.h5'
filepath = 'saved_models/VTCNN2_wts_032918.h5'
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

score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
    
sess = tf.InteractiveSession()
K.set_session(sess)
x = tf.placeholder(tf.float32, (None, img_rows, img_cols))
y = tf.placeholder(tf.float32, (None, nb_classes))
eps = tf.placeholder(tf.float32, ())
sess.run(tf.global_variables_initializer())     

def _model_fn(x, logits=False):
    ybar = model(x)
    #logits_,  = ybar.op.inputs
    logits_, tmp  = ybar.op.inputs
    if logits:
        return ybar, logits_
    return ybar

x_adv = fgsm(_model_fn, x, epochs=4, eps=0.01)

max_X = np.max(X_test.flatten())
min_X = np.min(X_test.flatten())
X_test_scaled = (X_test-min_X)/(max_X-min_X)

nb_sample = X_test_scaled.shape[0]
nb_batch = int(np.ceil(nb_sample/batch_size_adv))
X_adv = np.empty(X_test_scaled.shape)
for batch in range(nb_batch):
    print('batch {0}/{1}'.format(batch+1, nb_batch), end='\r')
    start = batch * batch_size_adv
    end = min(nb_sample, start+batch_size_adv)
    tmp = sess.run(x_adv, feed_dict={x: X_test_scaled[start:end],
                                     y: Y_test[start:end],
                                     K.learning_phase(): 0})
    X_adv[start:end] = tmp


#Low pass filter the adversarial examples
# import scipy as scp
# filt=np.reshape(.1*np.ones((10,)),[1, 1, 10])
# X_adv=scp.ndimage.filters.convolve(X_adv,filt)

X_adv = np.std(X_train.flatten())*(X_adv-np.mean(X_adv.flatten()))/np.std(X_adv.flatten())
np.save('X_adv_from_vtcnn2.npy', X_adv)

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
    model.load_weights(filepath)
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
fd = open('results_cnn2.dat','wb')
pickle.dump(acc, fd )
fd.close()
fd = open('results_cnn2_adversarial_white.dat','wb')
pickle.dump(acc_adv, fd )
fd.close()

