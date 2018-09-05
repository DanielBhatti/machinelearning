# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 21:30:42 2018

@author: bhatt
"""

import numpy
from keras.datasets import mnist # contains images of the numbers in forms of 
import matplotlib.pyplot as plt
import matplotlib.cm as cm # forms a color map

def main(image, im_filter):
    # image is the image to be interpreted, it will be a number specifiying a 28x28 image in an mnist array of 60,000 entries
    # im_filter is the filter we apply to the image
    # image will be in the form of a 28 x 28 matrix
    # im_filter will be in the form of a 3 x 3 matrix
    # im_filter is intended to change a given cell based on 3 x 3 neighbors
    
    
    im = X_train[image]
    # X_train will be a 60,000 entry array of 28 x 28 matrices
    
    width = im.shape[0]
    # gives the number of columns
    
    height = im.shape[1]
    # gives the number of rows
    
    imC = numpy.zeros((width-2, height-2))
    # creates matrix of zeroes with given dimensions
    # no clue why we're getting rid of 2 rows and columns
    
    
    for row in range(1,width-1):
        for col in range(1,height-1):
            for i in range(len(im_filter[0])):
                for j in range(len(im_filter)):
                    imC[row-1][col-1] += im[row-1+i][col-1+j] * im_filter[i][j]
                    # adds to each entry a multiplier of the 3x3 entries around it
                    # multiplied by the filter
                    
            if imC[row-1][col-1] > 255:
                imC[row-1][col-1] = 255
                # sets max value for intensity
                
            elif imC[row-1][col-1] < 0:
                imC[row-1][col-1] = 0
                # sets min value for intensity
                
    plt.imshow(im, cmap = cm.Greys_r)
    # plots original image
    plt.show()
    plt.imshow(imC/255, cmap = cm.Greys_r)
    # plots filtered image
    # don't know why we divide by 255.  Normalizing?
    
    plt.show()



if __name__ == '__main__':
    
    
    blur = [[1./9, 1./9, 1./9], [1./9, 1./9, 1./9], [1./9, 1./9, 1./9]]
    edges = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
    test = [[2,1,2],[1,2,1],[2,1,2]]
    
    
    
    entry = 8
    im_filter = test
    
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    main(entry, im_filter)