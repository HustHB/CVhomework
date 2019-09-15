# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:27:14 2019

@author: Dell
"""

from load_mnist import load_mnist
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
import math

def conv3(in_channel,out_channel):
    return nn.Sequential(nn.Conv2d(in_channel,out_channel,3,stride=1,padding=1))

class res_net(nn.Module):
    def __init__(self,in_channel,D):
        super(res_net,self).__init__()
        self.layers1=nn.Sequential(conv3(in_channel,D),
                                  nn.BatchNorm2d(D),
                                  nn.ReLU(inplace=True),
                                  conv3(D,D),
                                  nn.BatchNorm2d(D),
                                  nn.ReLU(inplace=True),
                                  )
        self.branch=conv3(in_channel,D)
        self.maxpool=nn.Sequential(nn.MaxPool2d(2))
        
    def forward(self,x):
        out=self.layers1(x)+self.branch(x)
        out=self.maxpool(out)
        return out

class my_net(nn.Module):
    def __init__(self,in_channel=1):
        super(my_net,self).__init__()
        self.in_channel=in_channel
        self.layer=nn.Sequential(res_net(self.in_channel,16),
                                 res_net(16,32),
                                 nn.Conv2d(32,64,3,stride=2),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(3)
                                 )
        self.layer2=nn.Sequential(nn.Linear(64,10),
                                  nn.Softmax(1)
                                  )
    
    def forward(self,input):
        out=self.layer(input)
        out=out.view(-1,64)
        out=self.layer2(out)
        return out

#x=torch.randn(4,1,28,28)
#print(x.shape)
#a=my_net()
#ftrs=a(x)
#print(ftrs.shape)

class my_dataset(Dataset):
    def __init__(self,data,ctype='train',transform=None):
        super(my_dataset,self).__init__()
        self.transform=transform
        self.ctype=ctype
        self.data=data
        
    def __len__(self):
        if self.ctype=='train':
            return self.data[0].shape[0]
        else:
            return self.data[2].shape[0]
    
    def __getitem__(self,idx):
        if self.ctype=='train':
            image=self.data[0][idx]
            label=self.data[1][idx]
             
        else:
            image=self.data[2][idx]
            label=self.data[3][idx]
        
        if self.transform:
            image=self.transform(image)
        
        return {'image':image,'label':label}

class img_processing(object):
    def __init__(self):
        pass
    
    def __call__(self,x):
        '''
        x is numpy img with formation h*w
        return 1*h*w torch.tensor
        '''
        img=torch.from_numpy(x).unsqueeze(0)
        return img.float()
        
def get_loss_acc(soft,y):
    loss=0.
    acc=((torch.argmax(soft,dim=1)==y.long()).sum()).float()/y.shape[0]
    s=-torch.log(soft)
    loss=s[range(soft.shape[0]),y.long()].sum()
    return loss,acc

def init_weight(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        n=m.kernel_size[0]*m.kernel_size[1]*m.in_channels
        m.weight.data.normal_(0,math.sqrt(2./n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm')!=-1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear')!=-1:
        n=m.weight.size(1)
        m.weight.data.normal_(0,0.01)
        m.bias.data=torch.ones(m.bias.data.size())


if __name__=='__main__':
    data=load_mnist()
#    plt.imshow(data[0][0])
#    plt.show()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    img_p=img_processing()
    train_data_set=my_dataset(data,transform=img_p)
    test_data_set=my_dataset(data,ctype='test',transform=img_p)
    
    net=my_net()
    net.apply(init_weight)
    net=net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
    
    train_acc_list=list()
    test_acc_list=list()
    for m in range(100):
        train_dataloader=DataLoader(train_data_set,shuffle=True,batch_size=3000,num_workers=0)
        test_dataloader=DataLoader(test_data_set,shuffle=False,batch_size=2000,num_workers=0)
        
        train_iter=iter(train_dataloader)
        test_iter=iter(test_dataloader)
        
        #train
        for i,train_data in enumerate(train_iter):
            train_img,train_label=train_data['image'].to(device),train_data['label'].to(device)
            train_soft=net(train_img)
            loss,acc=get_loss_acc(train_soft,train_label)
            
            #optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_acc_list.append(acc.item())
        print(m+1,'epoch train','loss:',loss.item(),'acc:',acc.item())
        
        #test
        epoch_acc_list=[]
        epoch_loss_list=[]
        for test_data in test_iter:
            test_img,test_label=test_data['image'].to(device),test_data['label'].to(device)
            test_soft=net(test_img)
            loss,acc=get_loss_acc(test_soft,test_label)
            epoch_acc_list.append(acc.item())
            epoch_loss_list.append(loss.item())
            
        aver_test_loss=sum(epoch_loss_list)/len(epoch_loss_list)
        aver_test_acc=sum(epoch_acc_list)/len(epoch_acc_list)
        test_acc_list.append(aver_test_acc)
        print(m+1,'epoch test','loss:',aver_test_loss,'acc:',aver_test_acc)
    
    plt.figure(1)    
    plt.plot(range(0,100),train_acc_list,c='b',label='train_curve')
    plt.plot(range(0,100),test_acc_list,c='r',label='test_curve')
    plt.xlabel('epoch')
    plt.xlim(0,100)
    plt.ylabel('accuracy')
    plt.ylim(0)
    plt.title('train_test_curve')
    plt.legend(loc = 0)
    plt.show()
    
    #get wrong discrimination picture
    test_dataloader=DataLoader(test_data_set,shuffle=False,batch_size=2000,num_workers=0)
    test_iter=iter(test_dataloader)
    
    wrong_img=[]
    for test_data in test_iter:
        test_img,test_label=test_data['image'].to(device),test_data['label'].to(device)
        test_soft=net(test_img)
        l=torch.argmax(test_soft,dim=1)==test_label.long()
        wrong_img_seq=[i for i,j in enumerate(l,0) if not j]
        wrong_img+=[i.squeeze(0) for i in test_img[wrong_img_seq]]
    
    #plot wrong example 
    plt.figure(2)
    wrong_img_example=wrong_img[0].cpu()
    wrong_img_example=wrong_img_example.data.numpy()
    plt.imshow(wrong_img_example)
    plt.show()
            
        








