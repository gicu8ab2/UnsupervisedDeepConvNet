import os
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
import h5py
import pickle
from attacks.fgsm import fgsm
K.set_image_dim_ordering('th')

# Set params
nb_classes = 11
nb_channels = 1
nb_rows = 2
nb_cols = 128
epochs = 20     # number of epochs to train on
batch_size_adv = 64  # training batch size

### Load and prep the RadioML dataset #######
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

max_X = np.max(X_test.flatten())
min_X = np.min(X_test.flatten())
X_test_scaled = (X_test-min_X)/(max_X-min_X)
####################################################

### Use VT-CNN2 (LeNet-5) as adversarial generator #############################
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
model.add(Dense( len(classes), kernel_initializer='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
filepath = '../saved_models/VTCNN2_wts_032918.h5'
model.load_weights(filepath)


##### FGSM Stuff ###############
sess = tf.InteractiveSession()
K.set_session(sess)
x = tf.placeholder(tf.float32, (None, nb_rows, nb_cols))
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


#Save adversarial data
f=h5py.File('../data/X_adv_from_vtcnn2.h5','w')
f.create_dataset('X_adv',data=X_adv)
f.close()



