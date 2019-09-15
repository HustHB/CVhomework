import numpy as np

import sys
sys.path.append('mnist')
import mnist
import matplotlib.pyplot as plt

def load_mnist():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()
    
    #plt.imshow(train_images[0])
    #plt.show()
    return train_images,train_labels,test_images,test_labels
    
#    n_train, w, h = train_images.shape
#    X_train = train_images.reshape((n_train, w*h))
#    Y_train = train_labels
#    
#    n_test, w, h = test_images.shape
#    X_test = test_images.reshape((n_test, w*h))
#    Y_test = test_labels
#    
#    print(X_train.shape, Y_train.shape)
#    print(X_test.shape, Y_test.shape)
#    return X_train,Y_train,X_test,Y_test

if __name__=='__main__':
    X_train,Y_train,X_test,Y_test=load_mnist()