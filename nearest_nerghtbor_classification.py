# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:33:07 2018

@author: 25742
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

############################# Application 1: truncated svd vs optshrink algorithm #####################################
def Optshrink(Y,r):
    Y=np.mat(Y)
    U,s,V=np.linalg.svd(Y)
    m,n=Y.shape
    r=min(r,m,n)
    if m >= n:
        S=np.concatenate([np.diag(s[r:n]),np.zeros([(m-n),(n-r)])], axis=0)
    else:
        S=np.concatenate([np.diag(s[r:m]),np.zeros([(m-r),(n-m)])], axis=1)
    w=np.zeros(r)
    for k in range(0,r):
        D,Dder=D_transfrom_from_matrix(s[k],S)
        w[k]=-2*D/Dder
    Xh = U[:,0:r]*np.diag(w)*V[0:r:,]
    return Xh

def D_transfrom_from_matrix(z,X):
    X=np.mat(X)
    n,m=X.shape
    In=np.mat(np.diag(np.ones(n)))
    Im=np.mat(np.diag(np.ones(m)))
    D1=np.trace(z*(z*z*In-X*X.T)**(-1))/n
    D2=np.trace(z*(z*z*Im-X.T*X)**(-1))/m
    D=D1*D2
    D1_der=np.trace(-2*z*z*(z*z*In-X*X.T)**(-2)+(z*z*In-X*X.T)**(-1))/n
    D2_der=np.trace(-2*z*z*(z*z*Im-X.T*X)**(-2)+(z*z*Im-X.T*X)**(-1))/m
    D_der=D1*D2_der+D2*D1_der
    return D, D_der

def rgb_to_grey(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# truncated singular value decomposition
def truncated_svd(A,r):
    A=np.mat(A)
    U,s,V=np.linalg.svd(A)
    m,n=A.shape
    tsvd=U[:,0:r]*np.diag(s[0:r])*V[0:r:,]
    return tsvd

# compare the performance between the optshrink algorithm and truncated svd
def compare_opt_tsvd(pure_img ,noise_img, rank):
    error_opt=[]
    error_tsvd=[]
    for i in range(1, rank+1):
        opt_dnoise=Optshrink(noise_img, i)
        tsvd_dnoise=truncated_svd(noise_img, i)
        error_opt_value=np.linalg.norm(pure_img-opt_dnoise)
        error_opt.append(error_opt_value)
        error_tsvd_value=np.linalg.norm(pure_img-tsvd_dnoise)
        error_tsvd.append(error_tsvd_value)
    plt.figure(figsize=(8,6))
    plt.plot(range(1, rank+1), error_opt, c='blue', label='optshrink method')
    plt.plot(range(1, rank+1), error_tsvd, c='red', linestyle='--', label='truncated svd method')
    plt.xlabel("Rank of the matrix")
    plt.ylabel("Error")
    plt.title('Optshrink method vs Truncated SVD method')
    plt.legend(loc='lower right', shadow=True)
    plt.show()

# CODE FOR APPLICATION 1: 

'''img=mpimg.imread('mit_logo.png')
img_1=rgb_to_grey(img)
m,n=img_1.shape
# add the noise matrix, you can change the value: sqrt(0.1) as you want
img_n=img_1+np.sqrt(0.1)*np.random.randn(m, n)
# plot to show whihc method is better
compare_opt_tsvd(img_1, img_n, min(m,n))
# plot the noise image
plt.imshow(img_n, cmap=plt.get_cmap('gray'))
# plot the denoise image
plt.imshow(Optshrink(img_n, 4), cmap=plt.get_cmap('gray'))'''



################################### Application 2 : digit classification #################################################
# data exploration
def data_exploration(train_t):
    fig=plt.figure(figsize=(11,11))
    m,n,p=train_t.shape
    row=4
    column=5
    for i in range(1,row+1): 
        if  i%2 == 1:
            train=train_t+66*np.random.normal(np.zeros([m, n, p]))
            for j in range(0,column):
                digit=np.reshape(train[:,:,int(j+5*(i-1)/2)][:,1], [28,28])
                fig.add_subplot(row, column, 5*(i-1)+j+1, title='nosie plus digit ' + str(int(j+5*(i-1)/2)))
                plt.imshow(digit, cmap=plt.get_cmap('gray'))         
        if i%2 == 0:
            train=train_t
            for j in range(0,column):
                digit=np.reshape(train[:,:,int(j+5*(i-2)/2)][:,1], [28,28])
                fig.add_subplot(row, column, 5*(i-2)+j+1+5, title='original digit ' + str(int(j+5*(i-2)/2)))
                plt.imshow(digit, cmap=plt.get_cmap('gray'))
            

# function to change traindata (10,800,784) to (784,800,10)
def transformation_matrix(train):
    m,n,b=train.shape
    train_t=np.zeros([b,n,m])
    for i in range(0,m):
        for j in range(0,n):
            train_t[:,:,i][:,j]=train[i,:,:,][j,:]
    return train_t

# after model selection, we use the parameter r = 10. 
def nearest_ss_optshrink(train, ktrain): # ktrain = n
    n,N,d=train.shape
    U=np.zeros([n,ktrain,d])
    for j in range(0,d):
        Uj=np.linalg.svd(Optshrink(train[:,:,j],10))[0]
        U[:,:,j]=Uj[:,0:ktrain]
    return U

def nearest_ss(train, k): # ktrain = n
    n,N,d=train.shape
    U=np.zeros([n,k,d])
    for j in range(0,d):
        Uj=np.linalg.svd(train[:,:,j])[0]
        U[:,:,j]=Uj[:,0:k]
    return U

def classify_manylabels(test, U, k, test_label): 
    correst=0
    n,p=test.shape
    d=U.shape[2]
    err=np.zeros([d,p])
    for j in range(0,d):
        Uj=U[:,0:k-1,j]
        err[j,:]=np.sum(np.square(np.mat(test)-np.mat(Uj)*(np.mat(Uj).T*np.mat(test))),axis=0)
    label=np.argmin(err,axis=0)
    for i in range(0,len(label)):
        if label[i]==test_label[i]:
            correst+=1
    pcorrect=correst/len(label)
    return pcorrect

def accuary_plot(test,opt_U,orig_U,test_label,maxN):
    pcorrect_opt=[]
    pcorrect_orig=[]
    for i in range(0,maxN):
        pcorrect_opt.append(classify_manylabels(test, opt_U, i+1, test_label))
    max_position=np.argmax(pcorrect_opt)
    max_value=pcorrect_opt[max_position]
    for i in range(0,maxN):
        pcorrect_orig.append(classify_manylabels(test, orig_U, i+1, test_label))
    plt.figure(figsize=(8,6))
    plt.plot(range(1,maxN+1), pcorrect_opt, c='blue', label='noise digit')
    plt.plot(range(1,maxN+1), pcorrect_orig, c='orange',linestyle='--', label='pure digit')
    plt.axvline(x=max_position+1, c='green', linestyle='--')
    plt.scatter([max_position+1], [max_value],c='red')
    plt.xlabel("Rank of the matrix")
    plt.ylabel("Accuracy %")
    plt.title('best classification accuracy = ' + str(max_value))
    plt.legend(loc='lower right', shadow=True)
    plt.show()
    return pcorrect_opt

def parameter_selection_opt(train_t, test_t, test_label, maxN, noise):
    m,n,p=train_t.shape
    train_n=train_t-noise*np.random.normal(np.zeros([m,n,p]))
    a,b=test_t.shape
    test_n=test_t-66*np.random.normal(np.zeros([a,b]))
    best_pcorrect=[]
    for r in range(1, 21):
        train_U_o=nearest_ss_optshrink(train_n, r)
        pcorrect_opt=[]
        for i in range(0,maxN):
            pcorrect_opt.append(classify_manylabels(test_n, train_U_o, i+1, test_label))
        max_position_opt=np.argmax(pcorrect_opt)
        max_value_opt=pcorrect_opt[max_position_opt]
        best_pcorrect.append(max_value_opt)
    return best_pcorrect

# CODE FOR APPLICATION 2:    

'''
######################################### load data ###################################################################
traindata=h5py.File('train_digits.mat')
train=np.array(traindata["train_data"]) 
train_t=transformation_matrix(train)
testdata=h5py.File('test_digits.mat')
test_label=np.array(testdata["test_label"]) 
test_data=np.array(testdata["test_data"]) 
test_data=test_data.T 

######################################### data exploration: ###########################################################
data_exploration(train_t)

# model selection: the best parameter 'r' in Optshrink function is depend on the noise, thus if we change the noise parameter, 
we need to redo the parameter selection to find a new optimal r. in our code, we set noise = 100, and the best r is 10.

best_pcorrect=parameter_selection_opt(train_t, test_data, test_label, 50, 100)
r=np.argmax(best_pcorrect)

# after we find the r, we need to fix r, thus, in our code, the 'r' in Optshrink algorithm is 10. 

###################################### noise dataset: #################################################################
np.random.seed(10086)
m,n,p=train_t.shape
train_n=train_t-100*np.random.normal(np.zeros([m,n,p]))

# explaination: for testing our improved method, we dodn't add noise to the test data. This is because we want to test our denoised 
data to the pure data: for the nearest subspace method, we use pure training data and pure testing data, for the optshrink nss method
, we need to control only one thing can be changed, and this is noise-plus training data. Thus, we did't add noise to the testing data.    

#################### calculate train_U_opt, train_U_original, test the performance of our method ######################
train_U_opt=nearest_ss_optshrink(train_n, 784)
train_U_original=nearest_ss(train_t, 784)
accuracy_plot(test_data, train_U_opt, train_U_original, test_label, 784)
'''


    
