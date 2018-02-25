import numpy as np
import scipy as sp
from scipy import stats
from sklearn.mixture import GaussianMixture
import torch
from torch.autograd import Variable

from util import *

## fit the mixture model with a linear trend and a Gaussian mixture 
def ML_init(p,x,K,alpha=0.1,n_itr=100,h=None,verbose=False):
    if verbose:
        print('### ML initialization starts ###\n')
        
    np.random.seed(42)
    if len(x.shape)==1:
        d = 1
    else:
        d = x.shape[1]
    
    ## extract the alternative proportion
    _,t_BH = bh(p,alpha=alpha)
    x_alt  = x[p<t_BH]
    n_samp = x_alt.shape[0]   
    x_alt_ = np.reshape(x_alt,[n_samp,d])
    
    GMM = GaussianMixture(n_components=K,covariance_type='diag')
    GMM.fit(x_alt_)
    
    ## initialization
    w,w_old  = np.ones([K+1])/(K+1),np.zeros([K+1])
    a        = ML_slope(x_alt)   
    mu,sigma = GMM.means_,GMM.covariances_**0.5
    w_samp   = np.zeros([K+1,n_samp],dtype=float)
    i        = 0
        
    while np.linalg.norm(w-w_old)>1e-3 and i<n_itr:        
        ## E step       
        w_old = w
        w_samp[0,:] = w[0]*f_slope(x_alt,a)
        for k in range(K):
            w_samp[k+1,:] = w[k+1]*f_bump(x_alt,mu[k],sigma[k])
        w_samp = w_samp/np.sum(w_samp,axis=0)
        
        ## M step
        a = ML_slope(x_alt,w_samp[0,:])
        for k in range(K):
            mu[k],sigma[k]=ML_bump(x_alt,w_samp[k+1,:])
        w = np.mean(w_samp,axis=1)
           
        i += 1

    if i >= n_itr:
        print('!!! the model does not converge !!!')
    
    if d==1:
        plt.figure()
        plt.hist(x_alt,bins=50,weights=1/n_samp*50*np.ones([n_samp]))
        temp = np.linspace(0,1,101)
        plt.plot(temp,f_all(temp,a,mu,sigma,w))
        plt.show()
    
    if verbose:
        print('## Learned parameters: ##')
        print('Slope: w=%s, a=%s'%(str(w[0]),str(a)))
        for k in range(K):
            print('Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w[k+1]),mu[k],sigma[k]))
        print('\n')
        
        t = w[0]*f_slope(x,a)
        for k in range(K):
            t += w[k+1]*f_bump(x,mu[k],sigma[k])
        t = rescale_mirror(t,p,alpha)
        if h is not None:
            result_summary(p<t,h)   
    return w,a,mu,sigma

## sub_routines for ML_init_1d
## fitting the slope function a/(e^a-1) e^(ax), defined over [0,1]
def ML_slope(x,w=None):
    def ML_slope_1d(x,w=None):
        if w is None:
            w = np.ones(x.shape[0])
        t = np.sum(w*x)/np.sum(w) ## sufficient statistic
        a_u=100
        a_l=-100

        ## binary search 
        while a_u-a_l>0.1:
            a_m  = (a_u+a_l)/2
            a_m += 1e-2*(a_m==0)
            if (np.log((np.exp(a_m)-1)/a_m)+0.05*a_m**2)/a_m<t:
                a_l = a_m
            else: 
                a_u = a_m

        return (a_u+a_l)/2
    if len(x.shape)==1:
        return ML_slope_1d(x,w)
    else:
        a = np.zeros(x.shape[1],dtype=float)
        for i in range(x.shape[1]):
            a[i] = ML_slope_1d(x[:,i],w)
        return a

def f_slope(x,a):
    if len(x.shape)==1: # 1d case 
        return a/(np.exp(a)-1)*np.exp(a*x)
    else:
        f_x = np.ones([x.shape[0]],dtype=float)
        for i in range(x.shape[1]):
            f_x *= a[i]/(np.exp(a[i])-1)*np.exp(a[i]*x[:,i])
        return f_x

## fitting the Gaussian bump function: (using ML, but did not correct for the finite interval)
def ML_bump(x,w=0):
    def ML_bump_1d(x,w=None):
        if w is None:
            w = np.ones(x.shape[0])
        mu    = np.sum(x*w)/np.sum(w)
        sigma = np.sqrt(np.sum((x-mu)**2*w)/np.sum(w))
        return mu,sigma
    
    if len(x.shape)==1:
        return ML_bump_1d(x,w)
    else:
        mu    = np.zeros(x.shape[1],dtype=float)
        sigma = np.zeros(x.shape[1],dtype=float)
        for i in range(x.shape[1]):
            mu[i],sigma[i] = ML_bump_1d(x[:,i],w)
        return mu,sigma
    
def f_bump(x,mu,sigma):
    def f_bump_1d(x,mu,sigma): ## correct for the finite interval issue
        pmf = sp.stats.norm.cdf(1,loc=mu,scale=sigma)-sp.stats.norm.cdf(0,loc=mu,scale=sigma)
        return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/2/sigma**2)/pmf
    
    if len(x.shape)==1:
        return f_bump_1d(x,mu,sigma)
    else:
        f_x = np.ones([x.shape[0]],dtype=float)
        for i in range(x.shape[1]):
            f_x *=  f_bump_1d(x[:,i],mu[i],sigma[i])
        return f_x
            

def f_all(x,a,mu,sigma,w):
    f = w[0]*f_slope(x,a)
    if len(x.shape) == 1:
        for k in range(1,w.shape[0]):
            f += w[k]*f_bump(x,mu[k-1],sigma[k-1])
    else:
        for k in range(1,w.shape[0]):
            f += w[k]*f_bump(x,mu[k-1,:],sigma[k-1,:])
            print(f[0:5])
    return f


## prim_fdr
def PrimFDR(p,x,K=2,alpha=0.1,n_itr=5000,h=None,verbose=False):    
    torch.manual_seed(42)
    d=1 if len(x.shape)==1 else x.shape[1]
    
    _,t_bh   = bh(p,alpha=alpha)
    lambda0  = 1/t_bh
    lambda1  = 10/alpha
    loss_rec = np.zeros([n_itr],dtype=float)
    n_samp   = x.shape[0]
    
    if verbose:
        print('### Parameters')
        print('n_itr=%s, n_samp=%s, lambda0=%s, lambda1=%s'%(str(n_itr),str(n_samp),str(lambda0),str(lambda1)))
    
    
    ## calculating the initialization value    
    w_init,a_init,mu_init,sigma_init = ML_init(p,x,K,alpha=alpha,verbose=True) 

    ## transforming the parameters in the probalistic model to the threshold
    a     = a_init
    b     = np.log(w_init[0]*(a_init/(np.exp(a_init)-1)).prod())    
    w     = np.log(w_init[1:]/((2*np.pi)**(d/2)*sigma_init.prod(axis=1)))
    mu    = mu_init 
    sigma = 2*sigma_init**2
    
    t = t_cal(x,a,b,w,mu,sigma)
    
    gamma=rescale_naive(t,p,alpha*0.9)
    b += np.log(gamma)
    w += np.log(gamma)    
    
    if verbose:
        t = rescale_mirror(t,p,alpha)
        if h is not None:
            print('## Summary before optimization ##')
            result_summary(p<t,h)
    
    ## initialization  
    lambda0 = Variable(torch.Tensor([lambda0]),requires_grad=False)
    lambda1 = Variable(torch.Tensor([lambda1]),requires_grad=False)
    p       = Variable(torch.from_numpy(p).float(),requires_grad=False)
    x       = Variable(torch.from_numpy(x).float(),requires_grad=False)
    if d == 1:
        a = Variable(torch.Tensor([a]),requires_grad=True)
    else:
        a = Variable(torch.Tensor(a),requires_grad=True)
        
    b       = Variable(torch.Tensor([b]),requires_grad=True)
    w       = Variable(torch.Tensor(w),requires_grad=True)
    mu      = Variable(torch.Tensor(mu),requires_grad=True)
    sigma   = Variable(torch.Tensor(sigma),requires_grad=True)    
    
    ## 
    if verbose:
        print('### initialization value ###')
        print ('Slope: a=%s, b=%s'%(str(a.data.numpy()),str(b.data.numpy())))
        for k in range(K):
            print('Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w.data.numpy()[k]),mu.data.numpy()[k],sigma.data.numpy()[k]))
        #print ('w0: %s, mu0: %s, sigma0: %s'%(str(w.data.numpy()[0]),str(mu.data.numpy()[0]),str(sigma.data.numpy()[0])))
        #print ('w1: %s, mu1: %s, sigma1: %s\n'%(str(w.data.numpy()[1]),str(mu.data.numpy()[1]),str(sigma.data.numpy()[1])))
        print('\n')
        
    optimizer = torch.optim.Adam([a,b,w,mu,sigma],lr=0.0005)
    optimizer.zero_grad()
    
    for l in range(n_itr):
        ## calculating the model
        optimizer.zero_grad()
        #t = torch.exp(x*a+b)       
        if d==1:
            t = torch.exp(x*a+b) 
        else:
            t = torch.exp(torch.matmul(x,a)+b)
        sigma = sigma.clamp(min=1e-6)
        for i in range(K):
            if d==1:
                t = t+torch.exp(w[i]-(x-mu[i])**2/sigma[i])
            else:
                t = t+torch.exp(w[i]-torch.matmul((x-mu[i,:])**2,1/sigma[i,:]))
        loss1 = -torch.mean(torch.sigmoid(lambda0*(t-p)))
        #loss2 = torch.exp(lambda1*(torch.mean(t)-alpha*torch.mean(torch.sigmoid(lambda0*(t-p)))))
        loss2 = lambda1*(torch.mean(t)-alpha*torch.mean(torch.sigmoid(lambda0*(t-p)))).clamp(min=0)
        loss  = loss1+loss2
        
        ## backprop
        loss.backward()
        optimizer.step()
        
        ## show the result 
        loss_rec[l] = loss.data.numpy()
        
        if verbose:
            if l%(int(n_itr)/5)==0:
                print('### iteration %s ###'%str(l))
                print('mean t: ', np.mean(t.data.numpy()))
                print('mean discovery: ', np.mean(t.data.numpy()>p.data.numpy()))
                print('loss1: ',loss1.data.numpy())
                print('loss2: ',loss2.data.numpy())
                print('n_rej: ',np.sum(t.data.numpy()>p.data.numpy()))
                print('Estimated FDP: %s'%str((torch.mean(t)/torch.mean(torch.sigmoid(lambda0*(t-p)))).data.numpy()))
                print('FDP: %s'%str( np.sum((h==0)*(p.data.numpy()<t.data.numpy()))/np.sum(p.data.numpy()<t.data.numpy())))               
                print ('Slope: a=%s, b=%s'%(str(a.data.numpy()),str(b.data.numpy())))
                for k in range(K):
                    print('Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w.data.numpy()[k]),mu.data.numpy()[k],sigma.data.numpy()[k]))

                
                if d==1:
                    plt.figure()
                    plot_t(t.data.numpy(),x.data.numpy())
                    plt.show()
                print('\n')
    
    
    plt.figure()
    plt.plot(loss_rec)
    plt.show()
    p = p.data.numpy()
    t = rescale_mirror(t.data.numpy(),p,alpha)
    n_rej=np.sum(p<t)    
        
    return n_rej,t

def rescale_naive(t,p,alpha):
    gamma_l = 0
    gamma_u = 1/np.mean(t)
    while gamma_u-gamma_l>1e-4 or (np.sum(t*gamma_m)/np.sum(p<t*gamma_m)>alpha):
        gamma_m = (gamma_l+gamma_u)/2
        if np.sum(t*gamma_m)/np.sum(p<t*gamma_m) < alpha:
            gamma_l = gamma_m
        else: 
            gamma_u = gamma_m
    return (gamma_u+gamma_l)/2

def rescale_mirror(t,p,alpha,return_gamma=0):
    gamma_l = 0
    gamma_u = 1/np.mean(t)
    gamma_m = (gamma_u+gamma_l)/2
    
    while gamma_u-gamma_l>1e-4 or (np.sum(p>1-t*gamma_m)/np.sum(p<t*gamma_m) > alpha):
        gamma_m = (gamma_l+gamma_u)/2
        #print(gamma_l,gamma_u,(np.sum(p>1-t*gamma_m))/np.sum(p<t*gamma_m))
        if (np.sum(p>1-t*gamma_m))/np.sum(p<t*gamma_m) < alpha:
            gamma_l = gamma_m
        else: 
            gamma_u = gamma_m
    if return_gamma==1:
        return t*(gamma_u+gamma_l)/2,(gamma_u+gamma_l)/2
    else:
        return t*(gamma_u+gamma_l)/2

def t_cal(x,a,b,w,mu,sigma):
    K = w.shape[0]
    t = np.exp(x.dot(a)+b)
    for i in range(K):
        if len(x.shape)==1:
            t += np.exp(w[i])*np.exp(-(x-mu[i])**2/sigma[i])
        else:
            t += np.exp(w[i])*np.exp(-np.sum((x-mu[i])**2/sigma[i],axis=1))
    return t

    
    