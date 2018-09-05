# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 21:14:56 2018

@author: bhatt

The purpose of the autoencoder is to take a set of data of memory N and map it into a new data set of memory M < N
Essentially, we are projecting the data set from one space to a subspace
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data/") # downloads all mnist data into directory
# images are all size 28 x 28
# mnist object has properties train and test in the form of numpy arrays

input_placeholder = tf.placeholder("float", shape=(None, 784))
# placeholder values will be assigned values later
# float specifies the type of value that will be passed
# shape specifies the dimensions
# None dimension represents variable length
# so we have an N x 784 matrix



weights = tf.Variable(tf.random_normal((784, 300), mean=0.0, stddev=1./784))
# variable values are, evidently, variable
# we must first initialize its values
# in this case we initialize using random Gaussian noise 
# the size is 784 x 300, the input to the number of hidden nodes



hidden_bias = tf.Variable(tf.zeros([300]))
visible_bias = tf.Variable(tf.zeros([784]))
# these will be the bias vectors for the hidden and visible nodes
# we are initializing them to have all entries be zero
# the vectors are n x 1


hidden_activation = tf.nn.sigmoid(tf.matmul(input_placeholder, weights) + hidden_bias)
#represents the activation of the hidden nodes
# matmul is a 2-argument function that outputs the dot product
# its arguments should be j x k and k x l
# this is p(h | V) = v * W + b
# v is the input
# W are the weights
# b is the hidden bias




visible_reconstruction = tf.nn.sigmoid(tf.matmul(hidden_activation, tf.transpose(weights)) + visible_bias)
# same as above, but we also add the visible bias before using the sigmoid function
# not sure what this is supposed to be though


final_hidden_activation =tf.nn.sigmoid(tf.matmul(visible_reconstruction, weights) + hidden_bias)
# don't know what this is supposed to represent



positive_phase = tf.matmul(tf.transpose(input_placeholder), hidden_activation)
# converting input into hidden layer, I think, to store more easily


negative_phase = tf.matmul(tf.transpose(visible_reconstruction),final_hidden_activation)
# reconverting the hidden layer back into a visible layer,I think





LEARING_RATE = 0.01
weight_update = weights.assign_add(LEARING_RATE * (positive_phase - negative_phase))



visible_bias_update = visible_bias.assign_add(LEARING_RATE * tf.reduce_mean(input_placeholder - visible_reconstruction, 0))
hidden_bias_update = hidden_bias.assign_add(LEARING_RATE * tf.reduce_mean(hidden_activation - final_hidden_activation, 0))
# how much we want to adjust the visible and hidden biases by



train_op = tf.group(weight_update, visible_bias_update,hidden_bias_update)
loss_op = tf.reduce_sum(tf.square(input_placeholder - visible_reconstruction))
# tf.group groups operations together as one operation
# the loss op will tell us how well we are training our neural net

session = tf.Session()
session.run(tf.initialize_all_variables())




current_epochs = 0
for i in range(300):
    total_loss = 0
    while mnist.train.epochs_completed == current_epochs:
        batch_inputs, batch_labels = mnist.train.next_batch(100)
        reconstruction_loss = session.run([train_op, loss_op], feed_dict={input_placeholder: batch_inputs})
        total_loss += reconstruction_loss[1]
    print("epochs %s loss %s" % (current_epochs, reconstruction_loss))
    current_epochs = mnist.train.epochs_completed


reconstruction = session.run(visible_reconstruction, feed_dict={input_placeholder:[mnist.train.images[0]]})

recarray = []

for i in range(0,28):
    recarray.append([])

for j in range(0,28):
    for i in range(0,28):
        recarray[j].append(reconstruction[0][28*j + i])


plt.imshow(recarray, cmap = 'hot', interpolation='nearest')