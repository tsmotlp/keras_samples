# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:46:57 2018

@author: tsmotlp
"""

#import libs
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

#get dataset
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

#data preprocessing
xtrain = xtrain.reshape(-1, 1, 28, 28)  #-1 list的个数, 1:channel
xtest = xtest.reshape(-1, 1, 28, 28)
ytrain = np_utils.to_categorical(ytrain, num_classes = 10)
ytest = np_utils.to_categorical(ytest, num_classes = 10)

#create model
model = Sequential()

#conv layer 1, output shape(32, 28, 28)
model.add(Convolution2D(
        nb_filter = 32,
        nb_row = 5,
        nb_col = 5,
        border_mode = 'same',   #padding method
        input_shape = (1, 28, 28)
        ))
model.add(Activation('relu'))
#pooling layer 1, output shape(32, 14, 14)
model.add(MaxPooling2D(
        pool_size = (2, 2),
        strides = (2, 2),
        border_mode = 'same'
        ))

#conv layer 2, output shape(64, 28, 28)
model.add(Convolution2D(32, 5, 5, border_mode = 'same'))
model.add(Activation('relu'))

#pooling layer 2, output shape(64, 7, 7)
model.add(MaxPooling2D((2, 2), (2, 2), border_mode = 'same'))

#fully-connected layer 1, input_shape(64*7*7), output_shape(1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

#fully-connected layer 2, output_shape(10 classes)
model.add(Dense(10))
model.add(Activation('softmax'))

#define the optimizer
adam = Adam(lr = 0.001)

#compile
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#train
print('Training...')
model.fit(xtrain, ytrain, batch_size = 32, epochs = 1)

#testing
print('Testing...')
loss, accuracy = model.evaluate(xtest, ytest)

print('loss:',loss)
print('accuracy:',accuracy)



