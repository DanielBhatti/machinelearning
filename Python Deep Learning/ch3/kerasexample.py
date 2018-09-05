# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 00:46:31 2018

@author: bhatt
"""


import winsound as sound
import time as time
from keras.datasets import mnist # dataset containing 70 000 handwritten digits as images
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main(X_train, X_test, Y_train, Y_test):
    model = Sequential() # model implemented in keras
    model.add(Dense(hidden_neurons, input_dim=input_size)) #specifies number of hidden and input neurons.  Input to first later
    model.add(Activation('sigmoid')) # specifies the activation function.  Input to first layer
    model.add(Dense(classes, input_dim=hidden_neurons)) # specifies number of output neurons and hidden neurons.  First layer to output
    model.add(Activation('softmax')) # specifies activation function.  First layer to output layer
    
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
    # Keras allows us to specify the cost function and its optimization - we more or less will take on only the default values
    # we optimize using sgd (stochastic gradient descent) 
    
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1)
    # fits the model based on our training data.  verbose lets us follow our progress
    
    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test accuracy:', score[1])
    
    weights = model.layers[0].get_weights() # gives us weights of inputs to first layer
    
    w = weights[0].T
    print(weights)
    for neuron in range(hidden_neurons):
        plt.imshow(np.reshape(w[neuron], (28, 28)), cmap = cm.Greys_r)
        plt.savefig("weights/" + str(neuron))



#def interpret(image.png):
    # open image
    # convert image to 27 x 27 array of intensity values
    # apply weights from weight matrix to this matrix
    # interpret value of output 


start_time = time.time()






(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# loads training and test data.  X data is a set of arrays (where each array contains 28 arrays containing arrays with 28 numerical entries)
# that is, it is an array where each element is a 28 x 28 matrix
# where each array contains the intensity of each pixel
# Y data contains the value of the number for each respective array



X_train = X_train.reshape(60000,784) # reshapes each matrix in array into a 784 entry 1-vector
X_test = X_test.reshape(10000,784)


classes = 10 # number of outputs (i.e. the digits 0 to 9)
Y_train = np_utils.to_categorical(Y_train, classes)
Y_test = np_utils.to_categorical(Y_test, classes)
# converts each digit in the array into a 10-vector with 1 giving digit value
# e.g. 4 --> (0,0,0,0,1,0,0,0,0,0)
# 9 --> (0,0,0,0,0,0,0,0,0,1)

input_size = 784 # size of each mnist image (the 28 x 28 array converted to a 1 x 784 vector)
batch_size = 100 # ???  Sample for how many samples to test each epoch?
hidden_neurons = 100 # number of neurons in hidden layer
epochs = 5
main(X_train, X_test, Y_train, Y_test)




# 10 17.4
# 20 44
# 100 170
# 1000 1491




total_time = time.time() - start_time
print(total_time)




duration = 2000  # millisecond
freq = 340  # Hz
sound.Beep(freq, duration)
