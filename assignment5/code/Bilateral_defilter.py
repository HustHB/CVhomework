# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:49:00 2019

@author: Dell
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def gauss(l,sigma=1):
    a=(2*np.pi*sigma**2)**1/2
    b=np.exp(-l**2/(2*sigma**2))
    return b/a

def load_img(root):
    img=Image.open(root)
    return img

def generate_gauss_kernel_img(img_block,sigma1=2,sigma2=0.1,kernel_size=3):
    assert isinstance(kernel_size,int)
    kernel_size=(kernel_size,kernel_size)
    
    img_block=img_block.astype(float)
    
    gl=np.zeros(kernel_size)
    gc=np.zeros(kernel_size)
    h_center=round((kernel_size[0]+1)/2-1)
    w_center=round((kernel_size[1]+1)/2-1)
    
    #generate distance gauss
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            l=(i-h_center)**2+(j-w_center)**2
            gl[i,j]=gauss(l,sigma1)
            l=(img_block[i,j]-img_block[h_center,w_center])**2
            gc[i,j]=gauss(l,sigma2)
    
    a=int(round(((gl*gc*img_block).sum()/((gl*gc).sum()))))
    return a

def bilateral_filter(img,sigma1=2,sigma2=0.1,kernel_size=3,stride=1):
    if (img.shape[0]-kernel_size)%stride!=0 or (img.shape[1]-kernel_size)%stride!=0:
        raise Exception('kernel_size or stride can\'t deal img exactly ')
    h_out_size=round((img.shape[0]-kernel_size)/stride+1)
    w_out_size=round((img.shape[1]-kernel_size)/stride+1)
    
    out_img=np.zeros((h_out_size,w_out_size,img.shape[2]),dtype=np.int)
    for channel in range(img.shape[2]):
        for w in range(w_out_size):
            for h in range(h_out_size):
                img_block=img[h:h+kernel_size,w:w+kernel_size,channel]
                out_img[h,w,channel]=generate_gauss_kernel_img(img_block,sigma1=sigma1,sigma2=sigma2,kernel_size=kernel_size)

    return out_img


img=load_img('.\\th.jfif')
img=np.array(img)
noise=np.random.randint(0,60,img.shape,dtype=np.int)
img_with_noise=(img+noise).clip(0,255)

out_img=bilateral_filter(img_with_noise)

plt.figure(1,figsize=(15,45))
plt.subplot(1,3,1)
plt.imshow(img)
plt.title('original img')
plt.subplot(1,3,2)
plt.imshow(img_with_noise)
plt.title('img with noise')
plt.subplot(1,3,3)
plt.imshow(out_img)
plt.title('img after filtering')
