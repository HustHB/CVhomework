# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:05:04 2019

@author: Dell
"""

from my_net import my_network
from load_mnist import load_mnist
import numpy as np

X_train,Y_train,X_test,Y_test=load_mnist()
X_subtrain=X_train[:1000]
Y_subtrain=Y_train[:1000]

#test lamda and epsilon(粗略估计范围版本)
#max_count=50
#for i in range(max_count):
#    lamda=10**np.random.uniform(-7,5)
#    epsilon=10**np.random.uniform(-3,-6)
#    
#    net=my_network(lamda=lamda,epsilon=epsilon)
#    for j in range(20):
#        loss,y_pred=net.forward(X_subtrain,Y_subtrain)
#        acc=(y_pred==Y_subtrain).sum()/Y_subtrain.shape[0]
#        dW1,db1,dW2,db2,dW3,db3,dW4,db4=net.backward()
#        net.update([dW1,db1,dW2,db2,dW3,db3,dW4,db4],mode='sgd')
#    
#    #testing
#    loss,y_pred=net.forward(X_test[:1000],Y_test[:1000])
#    test_acc=(y_pred==Y_test[:1000]).sum()/Y_subtrain.shape[0]
#    print('test_acc:',test_acc,'lamda:',lamda,'epsilon:',epsilon,'({}/50)'.format(i+1))

#test lamda and epsilon(精密估计范围版本)
max_count=50
for i in range(max_count):
    lamda=10**np.random.uniform(-7,-5)
    epsilon=10**np.random.uniform(-5,-4)
    
    net=my_network(lamda=lamda,epsilon=epsilon)
    for j in range(20):
        loss,y_pred=net.forward(X_subtrain,Y_subtrain)
        acc=(y_pred==Y_subtrain).sum()/Y_subtrain.shape[0]
        dW1,db1,dW2,db2,dW3,db3,dW4,db4=net.backward()
        net.update([dW1,db1,dW2,db2,dW3,db3,dW4,db4],mode='sgd')
    
    #testing
    loss,y_pred=net.forward(X_test[:1000],Y_test[:1000])
    test_acc=(y_pred==Y_test[:1000]).sum()/Y_subtrain.shape[0]
    print('test_acc:',test_acc,'lamda:',lamda,'epsilon:',epsilon,'({}/50)'.format(i+1))
    