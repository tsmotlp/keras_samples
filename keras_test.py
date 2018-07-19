# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 09:52:54 2018

@author: tsmotlp
"""

#import library
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

#create some data
xdata = np.linspace(-1, 1, 1000)
np.random.shuffle(xdata)  #打乱顺序
ydata = 0.5 * xdata + 2 + np.random.normal(0, 0.05, (1000,))
#plot data
plt.scatter(xdata, ydata)
plt.show()

#divided raw_data into train_data and test_data
xtrain = xdata[:800]
xtest = xdata[800:]
ytrain = ydata[:800]
ytest = ydata[800:]

#build nueral network from 1st layer to last layer
model = Sequential()    #create a sequential model
model.add(Dense(output_dim = 1, input_dim = 1))

#define the optimizer
adam = Adam(lr = 0.01)

#choose loss function and optimizing method
model.compile(loss = 'mse', optimizer = adam) 

#training
print('Training...')
model.fit(xtrain, ytrain, batch_size = 20, epochs = 15)
'''
for step in range(501):
    cost = model.train_on_batch(xtrain,ytrain)
    if(step % 100 == 0):
        print('training cost',cost)
'''
#testing
print('Testing...')
cost = model.evaluate(xtest, ytest, batch_size = 200)
print('testing cost',cost)
W,b = model.layers[0].get_weights()
print('Weights:',W,'\n', 'bias:',b)

#plotting the prediction
ypred = model.predict(xtest)
plt.scatter(xtest, ytest)
plt.plot(xtest, ypred, 'r-')
plt.show()


    