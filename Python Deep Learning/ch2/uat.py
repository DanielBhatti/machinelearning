# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 17:46:34 2018

@author: bhatt
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.neural_network.multilayer_perceptron import MLPClassifier


def tanh(x):
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))

def tanhprime(x):
    return (1 + tanh(x))*(1 - tanh(x))

class NeuralNetwork:
    def __init__(self, net_arch): # net_arch is the number of neurons in each layer
        self.activity = tanh
        self.activity_derivative = tanhprime
        self.layers = len(net_arch) # total number of layers
        self.steps_per_epoch = 10000
        self.arch = net_arch
        
        self.weights = []
        
        for layer in range(self.layers - 1): # setting up weight for each layer
            w = 2 * np.random.rand(net_arch[layer] + 1, net_arch[layer+1]) - 1 # initialization of the weights
            self.weights.append(w)
            
            
    def fit(self,data,labels,learning_rate=0.1,epochs=100): # will train neural network.  data is set of boolean 2-ples, labels is result of xor function on each pair
        ones = np.ones((1,data.shape[0])) # makes (n,m) matrix of 1's - will be bias layer
        Z = np.concatenate((ones.T,data),axis=1) # concatenates rows
        training = epochs * self.steps_per_epoch # total number of steps
        
        for k in range(training):
            if k % self.steps_per_epoch == 0:
                print('epochs: {}'.format(k/self.steps_per_epoch)) # updates progress at each epoch
                for s in data:
                    print(s,nn.predict(s)) # predict will be defined later
                    
                    sample = np.random.randint(data.shape[0]) # random data point selected
                    y = [Z[sample]]
                    
                    for i in range(len(self.weights) - 1):
                        activation = np.dot(y[i], self.weights[i])
                        activity = self.activity(activation)
                        activity = np.concatenate((np.ones(1),np.array(activity)))
                        y.append(activity)
                        
                    activation = np.dot(y[-1], self.weights[-1])
                    activity = self.activity(activation)
                    y.append(activity)
                        
                    
                    
                    error = labels[sample] - y[-1]
                    delta_vec = [error * self.activity_derivative(y[-1])]
                    
                    for i in range(self.layers - 2,0,-1):
                        error = delta_vec[-1].dot(self.weights[i][1:].T)
                        error = error * self.activity_derivative(y[i][1:])
                        delta_vec.append(error)
                    delta_vec.reverse()
                    
                    for i in range(len(self.weights)):
                        layer = y[i].reshape(1,nn.arch[i]+1)
                        delta = delta_vec[i].reshape(1,nn.arch[i+1])
                        self.weights[i] += learning_rate * layer.T.dot(delta)
                        
                        
    def predict(self, x):
        val = np.concatenate((np.ones(1).T, np.array(x)))
        for i in range(0,len(self.weights)):
            val = self.activity(np.dot(val,self.weights[i]))
            val = np.concatenate((np.ones(1).T,np.array(val)))
        
        return val[1]

if __name__ == '__main__':
    np.random.seed(100)
    
    nn = NeuralNetwork([2,2,1])
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    nn.fit(X,y,learning_rate=0.1,epochs=1000)
    
    print("Final prediction")
    
    for s in X:
        print(s,nn.predict(s))
        


mlp = MLPClassifier(random_state=1)
mlp.fit(X,y)

data = X    
markers = ('s', '*', '^')
colors = ('blue', 'green', 'red')
cmap = ListedColormap(colors)

x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
resolution = 0.01


x, y = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
Z = mlp.predict(np.array([x.ravel(), y.ravel()]).T)
Z = Z.reshape(x.shape)
plt.pcolormesh(x, y, Z, cmap=cmap)
plt.xlim(x.min(), x.max())