# %% 1
# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import copy
import time

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',  categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',  categories=categories)

num_train = len(newsgroups_train.data)
num_test  = len(newsgroups_test.data)

# max_features is an important parameter. You should adjust it.
vectorizer = TfidfVectorizer(max_features=500)

X = vectorizer.fit_transform( newsgroups_train.data + newsgroups_test.data )
X_train = X[0:num_train, :]
X_test = X[num_train:num_train+num_test,:]

Y_train = newsgroups_train.target
Y_test = newsgroups_test.target

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X_train, Y_train)

Y_predict = clf.predict(X_test)

print(Y_test)
print(Y_predict)

ncorrect = 0
for dy in  (Y_test - Y_predict):
	if 0 == dy:
		ncorrect += 1

print('text classification accuracy is {}%'.format(round(100.0*ncorrect/len(Y_test)) ) )

class network(object):
    def __init__(self,num_ftrs=500,num_cl=4,lamda1=25,lamda2=25):
        self.num_ftrs=num_ftrs
        self.num_cl=num_cl
        self.lamda1=lamda1
        self.lamda2=lamda2
        
        W1=np.random.randn(num_ftrs,50)
        b1=np.random.randn(1,50)
        W2=np.random.randn(50,num_cl)
        b2=np.random.randn(1,num_cl)
        
        self.model={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
        
    def forward(self,X,y):
        self.num_example=X.shape[0]
        self.X=X
        self.y=y
        self.label=np.zeros((X.shape[0],self.num_cl))
        for i,j in enumerate(y):
            self.label[i,j]=1
        
        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        self.z1=X.dot(W1)+b1
        self.a1=np.tanh(self.z1)
        self.z2=self.a1.dot(W2)+b2
        exp_scores=np.exp(self.z2)
        self.probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
        
        y_pred=np.argmax(self.probs,axis=1)
        loss=-(self.label*np.log(self.probs)).sum()/X.shape[0]\
        +self.lamda1*(W1**2).sum()+self.lamda2*(W2**2).sum()
        
        return loss,y_pred
    
    def backward(self):
        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        delta3=self.probs
        delta3[range(self.num_example),self.y]-=1
        dW2=(self.a1.T).dot(delta3)+self.lamda2*W2
        db2=np.sum(delta3,axis=0,keepdims=True)
        delta2=delta3.dot(W2.T)*(1-np.power(self.a1,2))
        dW1=((self.X).T).dot(delta2)+self.lamda1*W1
        db1=np.sum(delta2,axis=0)
        
        return dW1,db1,dW2,db2
    
    def update(self,dX,epsilon=1e-4):
        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        W1-=epsilon*dX[0]
        b1-=epsilon*dX[1]
        W2-=epsilon*dX[2]
        b2-=epsilon*dX[3]
        
        self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']=W1,b1,W2,b2



#my design
def relu(x):
    m=copy.copy(x)
    m[m<0]*=0.1
    return m

def drelu(x):
    m=copy.copy(x)
    m[m>0]=1
    m[m<=0]=0.1
    return m

class my_network(object):
    def __init__(self,num_ftrs=500,num_cl=4,lamda1=1e-4,lamda2=1e-4,lamda3=1e-4,lamda4=1e-4):
        self.num_ftrs=num_ftrs
        self.num_cl=num_cl
        self.lamda1=lamda1
        self.lamda2=lamda2
        self.lamda3=lamda3
        self.lamda4=lamda4
        
        W1=np.random.randn(num_ftrs,300)
        b1=np.random.randn(1,300)
        W2=np.random.randn(300,100)
        b2=np.random.randn(1,100)
        W3=np.random.randn(100,10)
        b3=np.random.randn(1,10)
        W4=np.random.randn(10,num_cl)
        b4=np.random.randn(1,num_cl)
        
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
        self.a1=relu(self.z1)
        self.z2=self.a1.dot(W2)+b2
        self.a2=relu(self.z2)
        self.z3=self.a2.dot(W3)+b3
        self.a3=relu(self.z3)
        self.z4=self.a3.dot(W4)+b4
        exp_scores=np.exp(self.z4)
        self.probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
        
        y_pred=np.argmax(self.probs,axis=1)
        loss=-(self.label*np.log(self.probs)).sum()/X.shape[0]\
        +self.lamda1*(W1**2).sum()+self.lamda2*(W2**2).sum()+self.lamda3*(W3**2).sum()+self.lamda4*(W4**2).sum()
        
        return loss,y_pred
    
    def backward(self):
        W1,b1,W2,b2,W3,b3,W4,b4=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2'],self.model['W3'],self.model['b3'],self.model['W4'],self.model['b4']
        delta5=self.probs
        delta5[range(self.num_example),self.y]-=1
        dW4=(self.a3.T).dot(delta5)+self.lamda4*W4
        db4=np.sum(delta5,axis=0,keepdims=True)
        delta4=delta5.dot(W4.T)*drelu(self.z3)
        dW3=((self.a2).T).dot(delta4)+self.lamda3*W3
        db3=np.sum(delta4,axis=0,keepdims=True)
        delta3=delta4.dot(W3.T)*drelu(self.z2)
        dW2=((self.a1).T).dot(delta3)+self.lamda2*W2
        db2=np.sum(delta3,axis=0,keepdims=True)
        delta2=delta3.dot(W2.T)*drelu(self.z1)
        dW1=((self.X).T).dot(delta2)+self.lamda1*W1
        db1=np.sum(delta2,axis=0,keepdims=True)
        
        
        return dW1,db1,dW2,db2,dW3,db3,dW4,db4
    
    def update(self,dX,epsilon=1e-5):
        W1,b1,W2,b2,W3,b3,W4,b4=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2'],self.model['W3'],self.model['b3'],self.model['W4'],self.model['b4']
        W1-=epsilon*dX[0]
        b1-=epsilon*dX[1]
        W2-=epsilon*dX[2]
        b2-=epsilon*dX[3]
        W3-=epsilon*dX[4]
        b3-=epsilon*dX[5]
        W4-=epsilon*dX[6]
        b4-=epsilon*dX[7]
        
        self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2'],self.model['W3'],self.model['b3'],self.model['W4'],self.model['b4']=W1,b1,W2,b2,W3,b3,W4,b4
        
if __name__=='__main__':
    #two layer neural net
    net=network()
    acc=0
    i=0
    while i<3000:
        loss,y_pred=net.forward(X_train,Y_train)
        acc=(y_pred==Y_train).sum()/Y_train.shape[0]
        
        if i%100==99:
           print(i+1,'\t','loss:',loss,'acc:',acc)
        dW1,db1,dW2,db2=net.backward()
        net.update([dW1,db1,dW2,db2])
        i+=1
        
    loss,y_pred=net.forward(X_test,Y_test)  
    print(y_pred[:20])
    ncorrect = 0
    for dy in  (Y_test - y_pred):
        if 0 == dy:
        	ncorrect += 1

    print('text classification accuracy is {}%'.format(round(100.0*ncorrect/len(Y_test))))


#four layer neural net    
    net=my_network()
    acc=0
    i=0
    while i<3000:
        loss,y_pred=net.forward(X_train,Y_train)
        acc=(y_pred==Y_train).sum()/Y_train.shape[0]                
        if i%100==99:
           print(i+1,'\t','loss:',loss,'acc:',acc)
        dW1,db1,dW2,db2,dW3,db3,dW4,db4=net.backward()
        net.update([dW1,db1,dW2,db2,dW3,db3,dW4,db4])
        i+=1
        
    loss,y_pred=net.forward(X_test,Y_test)  
    
    ncorrect = 0
    for dy in  (Y_test - y_pred):
        if 0 == dy:
        	ncorrect += 1

    print('text classification accuracy is {}%'.format(round(100.0*ncorrect/len(Y_test))))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        