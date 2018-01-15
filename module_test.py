import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

## generating the 1d toy example
def toy_data_1d(job_id=0,n_sample=10000,vis=0):
    if job_id == 0: # Gaussian mixtures     
        x   = np.random.uniform(-1,1,size=n_sample)
        pi1 = pi1_gen(x)
        p   = np.zeros(n_sample)
        
        # generating the hypothesis       
        h  = np.array((np.random.uniform(size=n_sample)<pi1),dtype=int)
        n0 = np.sum(h==0)
        n1 = np.sum(h==1)
        
        # generating the p-values 
        p[h==0] = np.random.uniform(size=n0)
        p[h==1] = np.random.beta(a=0.3,b=4,size=n1)
        
        if vis == 1:
            plt.figure(figsize=[12,5])
            plt.subplot(121)
            plot_pi1_1d(pi1_gen)
            plt.subplot(122)
            plot_data_1d(p,x,h)
            plt.legend()
            plt.show() 
        return p,x,h
# sub-routines for toy_data_1d
def pi1_gen(x): # need to be fixed 
    pi1=0.1*sp.stats.norm.pdf(x,loc=0.5,scale=0.2)+0.1*sp.stats.norm.pdf(x,loc=-0.8,scale=0.1)
    pi1+=0.1*(x+1) 
    return pi1

def plot_pi1_1d(pi1_gen):
    x_grid   = np.linspace(-1,1,200)
    pi1_grid = pi1_gen(x_grid)  
    plt.plot(x_grid,pi1_grid)
    plt.xlabel('covariate')
    plt.ylabel('alt distribution')
    plt.title('the alternative distribution')
    
def plot_data_1d(p,x,h):
    rnd_idx=np.random.permutation(p.shape[0])[0:1000]
    p = p[rnd_idx]
    x = x[rnd_idx]
    h = h[rnd_idx]
    plt.scatter(x[h==1],p[h==1],color='r',alpha=0.2,label='alt')
    plt.scatter(x[h==0],p[h==0],color='b',alpha=0.2,label='null')
    plt.xlabel('covariate')
    plt.ylabel('p-value')
    plt.title('hypotheses')    

## testing methods 
def bh(p,alpha=0.05):
    pass ## fix it

def storey_bh(p,alpha=0.05,lamb=0.6,vis=0):
    n_sample = p.shape[0]
    pi0_hat  = np.sum(p>lamb)/n_sample/(1-lamb)
    alpha   /= pi0_hat
    p_sort = sorted(p)
    n_rej = 0
    for i in range(n_sample):
        if p_sort[i] < i*alpha/n_sample:
            n_rej = i
    t_rej = p_sort[n_rej]
    if vis == 1:
        print("### Summary ###")
        print("method: Storey BH")
        print("# rejections: %s"%str(n_rej))
        print("rejection threshold: %s"%str(t_rej))
        print("null estimate: %s"%str(pi0_hat))
        print("### End Summary ###")
    return n_rej,t_rej,pi0_hat

# ancillary functions for PrimFDR
def plot_t(t,x):
    plt.scatter(x,t,alpha=0.2)
    plt.ylabel('t')
    plt.xlabel('x')



    