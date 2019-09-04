# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 21:01:32 2019

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)

num_observations=500
x1=np.random.multivariate_normal([0,0],[[1,0.75],[0.75,1]],num_observations)#mean,convariance matrix,number
x2=np.random.multivariate_normal([1,4],[[1,.75],[.75,1]],num_observations)

X=np.vstack((x1,x2)).astype(np.float32)
y=np.hstack((np.zeros(num_observations),np.ones(num_observations)))
#print(x1)
fig=plt.figure()
plt.scatter(X[:500,0],X[:500,1],c='pink')
plt.scatter(X[500:,0],X[500:,1],c='b')

X_constant=np.ones((2*num_observations,1))
X_final=np.concatenate((X_constant,X),axis=1)
w=np.zeros(3)#(3,1)

a=X_final@w

def my_sign(x):
    return 1 if x>0 else 0


def distance_sum_judge(w,X,y,threshold):
    s=(((X@w>0)-y)==0).sum()
    print('correct_rate:',s/X.shape[0])
    return s/X.shape[0]>1-threshold

def update_w(w,X,y,threshold=0.01,r=1):
    i=0
    while not distance_sum_judge(w,X,y,threshold):
        if i==1000:
            i=0
        pred=my_sign(X[i,:]@w)
        if pred!=y[i]:
            w=w+r*(y[i]-pred)*X[i,:]
            print('updated w:',w)
        i+=1
    return w

if __name__=='__main__':
    w=update_w(w,X_final,y,threshold=0.005)
    print('final_w',w)
    x_line=np.linspace(-5,5,num=1000)
    y_line=-(w[0]+w[1]*x_line)/w[2]
    plt.plot(x_line,y_line,c='r')