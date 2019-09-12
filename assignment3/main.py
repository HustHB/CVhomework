# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:46:40 2019

@author: Dell
"""

from my_net import my_network
from load_mnist import load_mnist
import time
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

X_train,Y_train,X_test,Y_test=load_mnist()

K=5#K fold cross validation
kf=KFold(n_splits=K)#K-Fold split

m=np.arange(len(X_train))
np.random.shuffle(m)
X_train=X_train[m]
Y_train=Y_train[m]

batch_size=2400
epoch_num=5


#main
best_acc=0
model={}
train_cv_acc_list=[0]
test_acc_list=[0]

for i in range(epoch_num):
    print(i+1,'epoch...')
    val_list=[]
    net=my_network(lamda=1.6*1e-6,epsilon=3.6*1e-5)
    for train,cv in kf.split(X_train):#train,cv 为train和cv的序号
        #split train_set and cross validation set
        X_train_set=X_train[train]
        Y_train_set=Y_train[train]
        X_cv_set=X_train[cv]
        Y_cv_set=Y_train[cv]
        
        
        #train
        print('training...')
        for k in range(20):#理论为20，为了加快速度，取5
            print(k+1,'batch...')
            X_subtrain=X_train_set[k*batch_size:(k+1)*batch_size]
            Y_subtrain=Y_train_set[k*batch_size:(k+1)*batch_size]
            for i in range(100): 
                loss,y_pred=net.forward(X_subtrain,Y_subtrain)
                acc=(y_pred==Y_subtrain).sum()/Y_subtrain.shape[0]
                if i%10==9:
                    print(i+1,'loss:',loss,'acc:',acc)
                dW1,db1,dW2,db2,dW3,db3,dW4,db4=net.backward()
                net.update([dW1,db1,dW2,db2,dW3,db3,dW4,db4],mode='sgd')
        
        #validation
        print('validating...')
        loss,y_pred=net.forward(X_cv_set,Y_cv_set)
        val_acc=(y_pred==Y_cv_set).sum()/Y_cv_set.shape[0]
        val_list.append(val_acc)
        print('validation acc:',val_acc)
    
    cv_aver_acc=sum(val_list)/len(val_list)
    print('cv_aver_acc:',cv_aver_acc)
    
    #testing
    print('testing...')
    loss,y_pred=net.forward(X_test,Y_test)
    test_acc=(y_pred==Y_test).sum()/Y_test.shape[0]
    print('test acc:',test_acc)
    
    if test_acc>best_acc:
        best_acc=test_acc
        model=net.model
   
    train_cv_acc_list.append(val_acc)
    test_acc_list.append(test_acc)
    
print('best_test_acc:',best_acc)
plt.figure(figsize=(32,20))
plt.plot(range(0,epoch_num+1),train_cv_acc_list,c='b',label='cross validation acc')
plt.plot(range(0,epoch_num+1),test_acc_list,c='r',label='test acc')
plt.xlim(0)
plt.ylim(0,1)
plt.title('cross validation acc and test acc curve')
plt.xlabel('epoch')
plt.ylabel('accuracy')


#X_subtrain=X_train_set[:2400]
#Y_subtrain=Y_train_set[:2400]
#
##training
#print('training...')
#t1=time.time()
#for i in range(1000):
#    loss,y_pred=net.forward(X_subtrain,Y_subtrain)
#    acc=(y_pred==Y_subtrain).sum()/Y_subtrain.shape[0]
#    print(i+1,'loss:',loss,'acc:',acc)
#    dW1,db1,dW2,db2,dW3,db3,dW4,db4=net.backward()
#    net.update([dW1,db1,dW2,db2,dW3,db3,dW4,db4],mode='sgd')
#
#print(time.time()-t1)
#
##testing
#print('testing...')
#loss,y_pred=net.forward(X_test,Y_test)
#test_acc=(y_pred==Y_test).sum()/Y_subtrain.shape[0]
#print('test acc:',test_acc)