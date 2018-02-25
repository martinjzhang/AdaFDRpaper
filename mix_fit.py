import numpy as np
import scipy as sp
from scipy import stats
from sklearn.mixture import GaussianMixture
import torch
from torch.autograd import Variable

from util import *

# first fit a slope, and the use the Gaussian to charecterize the bumps
# fix it 
def mixutre_fit(x,K,x_w=None,n_itr=100,verbose=False,debug=False):   
    d=1 if len(x.shape)==1 else x.shape[1]
    n_samp = x.shape[0]
    if x_w is None: x_w=np.ones([n_samp],dtype=float)
        
    ## initialization
    GMM = GaussianMixture(n_components=K,covariance_type='diag').fit(np.reshape(x,[n_samp,d]))
    w,w_old  = np.ones([K+1])/(K+1),np.zeros([K+1])
    a        = ML_slope(x)   
    mu,sigma = GMM.means_,GMM.covariances_**0.5
    w_samp   = np.zeros([K+1,n_samp],dtype=float)
    i        = 0
        
    while np.linalg.norm(w-w_old)>1e-3 and i<n_itr:        
        ## E step       
        w_old = w
        w_samp[0,:] = w[0]*f_slope(x,a)
        for k in range(K):
            w_samp[k+1,:] = w[k+1]*f_bump(x,mu[k],sigma[k])
        w_samp = w_samp/np.sum(w_samp,axis=0)*x_w
             
        ## M step
        w = np.mean(w_samp,axis=1) 
        a = ML_slope(x,w_samp[0,:])
        for k in range(K):
            if w[k+1]>1e-4: mu[k],sigma[k]=ML_bump(x,w_samp[k+1,:])               
        sigma = sigma.clip(min=1e-4)
        w[w<1e-4] = 0
        i += 1
        
        if i%10==0 and debug:
            print('## iteration %s ##'%str(i))
            print('Slope: w=%s, a=%s'%(str(w[0]),str(a)))
            for k in range(K):
                print('Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w[k+1]),mu[k],sigma[k]))
            print('\n')

    if i >= n_itr: print('!!! the model does not converge !!!')      
    
    if d==1 and verbose:
        plt.figure(figsize=[18,5])
        plt.subplot(121)
        temp_hist,_,_=plt.hist(x,bins=50,weights=1/n_samp*50*np.ones([n_samp]))
        temp = np.linspace(0,1,101)
        plt.plot(temp,f_all(temp,a,mu,sigma,w))
        plt.ylim([0,1.5*temp_hist.max()])
        plt.title('unweighted')
        plt.subplot(122)
        temp_hist,_,_=plt.hist(x,bins=50,weights=x_w/n_samp*50)
        temp = np.linspace(0,1,101)
        plt.plot(temp,f_all(temp,a,mu,sigma,w))
        plt.ylim([-1e-3,1.5*temp_hist.max()])
        plt.title('weighted')
        plt.show()   
    return w,a,mu,sigma 