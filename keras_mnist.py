# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:20:54 2018

@author: tsmotlp
"""

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


#download the mnist dataset
#xdata(60000,28x28), ydata(1000)
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

#data preprocessing
xtrain = xtrain.reshape(xtrain.shape[0], -1)/255  #normalization
xtest = xtest.reshape(xtest.shape[0], -1)/255    #normalization
#print('xtrain',xtrain)
#print('xtest',xtest)
ytrain = np_utils.to_categorical(ytrain, 10)
ytest = np_utils.to_categorical(ytest, 10)

#create the network
model = Sequential([
        Dense(32, input_dim = 784),
        Activation('relu'),
        Dense(10),
        Activation('softmax')
        ])

#define optimizer
rmsprop = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)

#compile
model.compile(
        optimizer = rmsprop,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])

#training
print('Training...')
model.fit(xtrain, ytrain, batch_size = 32, epochs = 2)

#testing
print('Testing...')
loss, accuracy = model.evaluate(xtest, ytest)

print('test loss:', loss)
print('test accuracy:', accuracy)
