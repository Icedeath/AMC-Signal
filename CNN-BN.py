from __future__ import print_function
import numpy
numpy.random.seed(1337)  # for reproducibility

import cPickle
import gzip
import os
import sys
import timeit
import h5py
import numpy
import scipy.io as sio 

from keras.callbacks import ModelCheckpoint  
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization as BN
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt

# There are 40 different classes
nb_classes = 10
nb_epoch = 1500
batch_size = 1000
length = 1500

num_classes = 10
num_snr = 21
num = Y_test.size/num_classes**2/num_snr
stride = Y_test.size/num_snr/num_classes

load = 0
loadfile = '10_-5_15_10W.h5'
savefile = '10_-5_15_10W.h5'
dataset = '10_-5_15_10W.mat'

# input image dimensions
# number of convolutional filters to use

def Net_model(lr=0.001,decay=1e-6,momentum=0.9):
    model = Sequential()
    model.add(Convolution2D(35, 1, 9,
                            border_mode='valid',
                            input_shape=(1, 1, length)))
    model.add(Activation(ELU(alpha=1.0)))
    
    model.add(Convolution2D(35, 1, 9))
    model.add(Activation(ELU(alpha=1.0)))
    
    model.add(MaxPooling2D(pool_size=(1, 4),strides = (1,3)))
    
    model.add(Convolution2D(40, 1, 9))
    model.add(Activation(ELU(alpha=1.0)))
    model.add(MaxPooling2D(pool_size=(1, 4),strides = (1,3)))
    
    model.add(Flatten())
    
    model.add(Dense(600,W_regularizer=l2(0.001))) #Full connection
    model.add(Dropout(0.2))
    model.add(Activation(ELU(alpha=1.0)))
    model.add(BN(epsilon=1e-06, mode=0, axis=1, momentum=momentum))
      
    model.add(Dense(200,W_regularizer=l2(0.001))) #Full connection
    model.add(Dropout(0.2))
    model.add(Activation(ELU(alpha=1.0)))
    model.add(BN(epsilon=1e-06, mode=0, axis=1, momentum=momentum)) 
    
    model.add(Dense(2,W_regularizer=l2(0.000))) #Full connection
    model.add(Dropout(0.2))
    model.add(Activation(ELU(alpha=1.0)))
    model.add(BN(epsilon=1e-06, mode=0, axis=1, momentum=momentum))
  
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])
    return model

def train_model(model,X_train,Y_train,X_val,Y_val):
    model.load_weights(loadfile)
    checkpointer =ModelCheckpoint(filepath=savefile, monitor='val_acc', verbose = 1,save_best_only=True)
    hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=2, validation_data=(X_val, Y_val),callbacks=[checkpointer])
    return hist.history

def test_model(model,X,Y):
	score = model.evaluate(X, Y, show_accuracy=True, verbose=0)
	return score
 
def accuracy(j):
    return [numpy.mean(numpy.equal(test_set_y[i*num + j*stride: (i+1)*num+j*stride],classes[i*num + j*stride: (i+1)*num+j*stride])) for i in xrange(num_classes)]

if __name__ == '__main__':
    '''
    train_set_x = numpy.float64(sio.loadmat(dataset, appendmat=False)['datasets'][0,0])
    train_set_y = numpy.int64(sio.loadmat(dataset, appendmat=False)['datasets'][0,1])-1

    
    test_set_x = numpy.float64(sio.loadmat(dataset, appendmat=False)['datasets'][1,0])
    test_set_y = numpy.int64(sio.loadmat(dataset, appendmat=False)['datasets'][1,1])-1
    test_set_y.shape = (max(test_set_y.shape),)

    temp_a,temp_b=train_set_x.shape
    temp_c,temp_d=test_set_x.shape
    '''
    
train_set_x = numpy.transpose(h5py.File(dataset)['train_x'].value)
train_set_y = numpy.transpose(h5py.File(dataset)['train_y'].value)-1
test_set_x = numpy.transpose(h5py.File(dataset)['test_x'].value)
test_set_y = numpy.transpose(h5py.File(dataset)['test_y'].value)-1
test_set_y.shape = (max(test_set_y.shape),)   
train_set_y.shape = (max(train_set_y.shape),)   
X_train = train_set_x.reshape(train_set_x.shape[0], 1, 1, train_set_x.shape[1])
X_test = test_set_x.reshape(test_set_x.shape[0], 1, 1, test_set_x.shape[1])
X_val = X_test

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'validate samples')
print(X_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(train_set_y, nb_classes)
Y_test = np_utils.to_categorical(test_set_y, nb_classes)
Y_val = Y_test
del train_set_x
del test_set_x
model=Net_model()
history = train_model(model,X_train,Y_train,X_val,Y_val)	

model.load_weights(loadfile)
classes_tr=model.predict_classes(X_train,verbose=0)
train_accuracy = numpy.mean(numpy.equal(train_set_y,classes_tr))
print("train accuarcy:",train_accuracy)

classes=model.predict_classes(X_test,verbose=0)
test_accuracy = numpy.mean(numpy.equal(test_set_y,classes))
print("test accuarcy:",test_accuracy)

train_loss = numpy.array(history['loss'])
val_loss = numpy.array(history['val_loss'])

a=numpy.arange(0,nb_epoch,1)
plt.plot(a,train_loss,label="train_loss",color="red",linewidth=2)
plt.plot(a,val_loss,label="val_loss",color="blue",linewidth=2)
plt.legend()
plt.show()

a=train_set_y.shape[0]/10
ac_train = [numpy.mean(numpy.equal(train_set_y[j*a:(j+1)*a],classes_tr[j*a:(j+1)*a])) for j in xrange(10)]
b=test_set_y.shape[0]/10
ac_test = [numpy.mean(numpy.equal(test_set_y[i*b:(i+1)*b],classes[i*b:(i+1)*b])) for i in xrange(10)]
ace = numpy.array([accuracy(j) for j in xrange(num_snr)])