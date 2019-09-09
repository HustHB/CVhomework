import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
import sklearn.datasets 
import sklearn.linear_model 
import matplotlib
import time
 
# Display plots inline and change default figure size 
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0) 
 

np.random.seed(3) 
X, y = sklearn.datasets.make_moons(200, noise=0.20) 
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral) 


# Train the logistic rgeression classifier 
clf = sklearn.linear_model.LogisticRegressionCV(cv=5) 
clf.fit(X, y) 
 

# Helper function to plot a decision boundary. 
# If you don't fully understand this function don't worry, it just generates the contour plot below. 
def plot_decision_boundary(pred_func): 
    # Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    # Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 

# Plot the decision boundary 
plot_decision_boundary(lambda x: clf.predict(x)) 
plt.title("Logistic Regression") 

plt.show()



#model={}
#x=np.random.randn(13,15)
def predict(model,x):
    W1,b1,W2,b2=model['W1'],model['b1'],model['W2'],model['b2']
    z1=x.dot(W1)+b1
    a1=np.tanh(z1)
    z2=a1.dot(W2)+b2
    exp_scores=np.exp(z2)
    probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    return np.argmax(probs,axis=1)

def get_acc(model,x,y):
    y_pred=predict(model,x)
    return (y_pred==y).sum()/x.shape[0]
#num_passes=10
#num_examples=100
#reg_lamda=0.5
#epsilon=0.8
#for i in range(0,num_passes):
#    #forward propagation
#    W1,b1,W2,b2=model['W1'],model['b1'],model['W2'],model['b2']
#    z1=X.dot(W1)+b1
#    a1=np.tanh(z1)
#    z2=a1.dot(W2)+b2
#    exp_scores=np.exp(z2)
#    probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
#
#    #backpropagation
#    delta3=probs
#    delta3[range(num_examples),y]-=1
#    dW2=(a1.T).dot(delta3)
#    db2=np.sum(delta3,axis=0,keepdims=True)
#    delta2=delta3.dot(W2.T)*(1-np.power(a1,2))
#    dW1=np.dot(X.T,delta2)
#    db1=np.sum(delta2,axis=0)
#    
#    #add regulation terms
#    dW2+=reg_lamda*W2
#    dW1+=reg_lamda*W1
#    
#    #Gradient descent parameter update
#    W1+=-epsilon*dW1
#    b1+=-epsilon*db1
#    W2+=-epsilon*dW2
#    b2+=-epsilon*db2
    
#while True:
#    data_batch=dataset.sample_data_batch()
#    loss=network.forward(data_batch)
#    dx=network.backward()
#    x+=-learning_rate*dx
    
class network(object):
    def __init__(self,ftrs_num,cl_num):
        '''
        input X:example_num*ftrs_num; there are cl_num different class
        '''
        self.ftrs_num=ftrs_num
        self.cl_num=cl_num
        
        W1=np.random.randn(ftrs_num,20)
        b1=np.random.randn(1,20)
        W2=np.random.randn(20,cl_num)
        b2=np.random.randn(1,cl_num)
        
        self.model={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    
    def forward(self,X,y):
        self.y=y
        self.example_num=X.shape[0]
        self.y_ext=np.zeros((self.example_num,self.cl_num))
        for i,j in enumerate(y,0):
            self.y_ext[i,j]=1 

        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        self.z1=X.dot(W1)+b1
        self.a1=np.tanh(self.z1)
        self.z2=self.a1.dot(W2)+b2
        self.exp_scores=np.exp(self.z2)
        self.probs=self.exp_scores/np.sum(self.exp_scores,axis=1,keepdims=True)
        loss=-(self.y_ext*np.log(self.probs)).sum()/self.example_num
        return loss
        
    
    def backward(self):
        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        delta3=self.probs
        delta3[range(self.example_num),self.y]-=1
        dW2=(self.a1.T).dot(delta3)
        db2=np.sum(delta3,axis=0,keepdims=True)
        delta2=delta3.dot(W2.T)*(1-np.power(self.a1,2))
        dW1=np.dot(X.T,delta2)
        db1=np.sum(delta2,axis=0)
        
        return dW1,db1,dW2,db2
    
    def update_net(self,dX,epsilon=0.001):
        '''
        dX:[dW1,db1,dW2,db2]
        '''
        W1,b1,W2,b2=self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']
        W1+=-epsilon*dX[0]
        b1+=-epsilon*dX[1]
        W2+=-epsilon*dX[2]
        b2+=-epsilon*dX[3]
        self.model['W1'],self.model['b1'],self.model['W2'],self.model['b2']=W1,b1,W2,b2
        
if __name__=='__main__':
    net=network(2,2)
    i=0
    best_acc=0
    best_model={}
    t=time.time()
    while True:
        loss=net.forward(X,y)
        acc=get_acc(net.model,X,y)
        if best_acc<acc:
            best_acc=acc
            best_model=net.model
        print('%d loss:'%(i+1),loss,' acc:',acc)
        if acc>0.99:
            break
        if i>100000:
            break
        dW1,db1,dW2,db2=net.backward()
        net.update_net([dW1,db1,dW2,db2])
        i+=1
    print('used time:',time.time()-t)    
    print('after %d iteration,best_acc: %f'%(i+1,best_acc),'\n','model:',best_model)    
    plot_decision_boundary(lambda x: predict(best_model,x))
        
    