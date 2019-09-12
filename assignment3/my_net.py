# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:07:16 2019

@author: Dell
"""

import numpy as np
import copy

def init_weight(in_channel,out_channel):
    w=np.random.randn(in_channel,out_channel)/np.sqrt(in_channel/2)
    return w

def relu(x):
    m=copy.copy(x)
    m[m<0]=0
    return m

def drelu(x):
    m=copy.copy(x)
    m[m>0]=1
    m[m<=0]=0
    return m

def batch_norm(x):
    pass

class my_network(object):
    def __init__(self,num_ftrs=784,num_cl=10,lamda=1e-4,epsilon=1e-5):
        self.num_ftrs=num_ftrs
        self.num_cl=num_cl
        self.lamda=lamda
        self.epsilon=epsilon
        self.cache=np.zeros(8)
        
        W1=init_weight(num_ftrs,300)
        b1=init_weight(1,300)
        W2=init_weight(300,100)
        b2=init_weight(1,100)
        W3=init_weight(100,10)
        b3=init_weight(1,10)
        W4=init_weight(10,num_cl)
        b4=init_weight(1,num_cl)
        
        self.model={'W1':W1,'b1':b1,'W2':W2,'b2':b2,'W3':W3,'b3':b3,'W4':W4,'b4':b4}
        
    def forward(self,X,y):
        self.num_example=X.shape[0]
        self.X=X
        self.y=y
        self.label=np.zeros((X.shape[0],self.num_cl))
        for i,j in enumerate(y):
            self.label[i,j]=1
        
        W1,b1,W2,b2,W3,b3,W4,b4=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2'],self.model['W3'],self.model['b3'],self.model['W4'],self.model['b4']
        self.z1=X.dot(W1)+b1
        self.a1=np.tanh(self.z1)
        self.z2=self.a1.dot(W2)+b2
        self.a2=relu(self.z2)
        self.z3=self.a2.dot(W3)+b3
        self.a3=relu(self.z3)
        self.z4=self.a3.dot(W4)+b4
        exp_scores=np.exp(self.z4)
        self.probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
        
        y_pred=np.argmax(self.probs,axis=1)
        loss=-(self.label*np.log(self.probs)).sum()/X.shape[0]\
        +self.lamda*((W1**2).sum()+(W2**2).sum()+(W3**2).sum()+(W4**2).sum())
        
        return loss,y_pred
    
    def backward(self):
        W1,b1,W2,b2,W3,b3,W4,b4=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2'],self.model['W3'],self.model['b3'],self.model['W4'],self.model['b4']
        delta5=self.probs
        delta5[range(self.num_example),self.y]-=1
        dW4=(self.a3.T).dot(delta5)+self.lamda*W4
        db4=np.sum(delta5,axis=0,keepdims=True)
        delta4=delta5.dot(W4.T)*drelu(self.z3)
        dW3=((self.a2).T).dot(delta4)+self.lamda*W3
        db3=np.sum(delta4,axis=0,keepdims=True)
        delta3=delta4.dot(W3.T)*drelu(self.z2)
        dW2=((self.a1).T).dot(delta3)+self.lamda*W2
        db2=np.sum(delta3,axis=0,keepdims=True)
        delta2=delta3.dot(W2.T)*(1-np.power(self.a1,2))
        dW1=((self.X).T).dot(delta2)+self.lamda*W1
        db1=np.sum(delta2,axis=0,keepdims=True)
        
        
        return dW1,db1,dW2,db2,dW3,db3,dW4,db4
    
    def update(self,dX,mode='sgd'):
        W1,b1,W2,b2,W3,b3,W4,b4=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2'],self.model['W3'],self.model['b3'],self.model['W4'],self.model['b4']
        if mode=='ada':
            #update cache
            for i in range(self.cache.shape[0]):
                self.cache[i]+=((dX[i])**2).sum()
                dX[i]/=(np.sqrt(self.cache[i])+1e-7)
        
        W1-=self.epsilon*dX[0]
        b1-=self.epsilon*dX[1]
        W2-=self.epsilon*dX[2]
        b2-=self.epsilon*dX[3]
        W3-=self.epsilon*dX[4]
        b3-=self.epsilon*dX[5]
        W4-=self.epsilon*dX[6]
        b4-=self.epsilon*dX[7]
        
        self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2'],self.model['W3'],self.model['b3'],self.model['W4'],self.model['b4']=W1,b1,W2,b2,W3,b3,W4,b4
