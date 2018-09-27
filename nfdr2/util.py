import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import logging

""" 
    basic functions
"""
def get_grid_1d(n_grid):    
    """
    return an equally spaced covariate covering the 1d space...

    Parameters
    ----------
    n_grid: int
        number of points 

    Returns
    -------
    (n,1) ndarray
    """
    
    x_grid = np.linspace(0,1,n_grid).reshape(-1,1)
    return x_grid

def get_grid_2d(n_grid):
    """
    return an equally spaced covariate covering the 2d space...

    Parameters
    ----------
    n_grid: int
        number of points 

    Returns
    -------
    (n,2) ndarray
    """
    temp = np.linspace(0,1,n_grid)
    g1,g2  = np.meshgrid(temp,temp)
    x_grid = np.concatenate([g1.reshape(-1,1),g2.reshape(-1,1)],axis=1)
    return x_grid

"""
    calculate the dimension-wise rank statistics
    # fix it: for discrete features, it may be nice to keep their values the same
    
    ----- input  -----
    x: an n*d array 
    
    ----- output -----
    ranks: an n*d array, column-wise rank of x
"""
def rank(x, continous_rank=True):
    """Calculate the dimension-wise rank statistics.
    
    Args:
        x ((n,d) ndarray): The covariates.
        continous_rank (bool): Indicate if break the same value by randomization.
    
    Returns:
        ranks ((n,d) ndarray): The column-wise rank of x
    """
    ranks = np.empty_like(x)
    n,d = x.shape
    for i in range(d):
        if continous_rank:           
            temp = x[:,i].argsort(axis=0)       
            ranks[temp,i] = np.arange(n)
        else:
            ranks[:,i] = rankdata(x[:,i])-1
    return ranks

def result_summary(pred, h=None, f_write=None, title=''):
    """ Summerize the result based on the predicted value and the true value
    
    Args:
        pred ((n,) ndarray): the testing result, 1 for alternative and 0 for null.
        h ((n,) ndarray): the true values.
        f_write (file handle)
        
    """
    if title != '':
        print('## %s'%title)
    print('# Num of discovery: %d'%np.sum(pred))
    if h is not None: 
        print("# Num of alternatives:",np.sum(h))
        print("# Num of true discovery: %d"%np.sum(pred*h))
        print("# Actual FDP: %0.3f"%(1-np.sum(pred * h) / np.sum(pred)))
    print('')    
    if f_write is not None:
        f_write.write('# Num of discovery: %d\n'%np.sum(pred))
        if h is not None:
            f_write.write("# Num of alternatives: %d\n"%np.sum(h))
            f_write.write("# Num of true discovery: %d\n"%np.sum(pred*h))
            f_write.write("# Actual FDP: %0.3f\n"%(1-np.sum(pred * h) / np.sum(pred)))
        f_write.write('\n')
    return

def print_param(a,mu,sigma,w):
    print('# w=%s'%w)
    print('# a=%s'%a)
    print('# mu=%s'%mu)
    print('# sigma=%s'%sigma)
    print('')

"""
    basic functions for visualization
""" 
def plot_x(x,vis_dim=None):
    if len(x.shape)==1:
        plt.hist(x,bins=50)
    else:
        if vis_dim is None: vis_dim = np.arange(x.shape[1])            
        for i,i_dim in enumerate(vis_dim):
            plt.subplot('1'+str(len(vis_dim))+str(i+1))
            plt.hist(x[:,i_dim],bins=50)
            plt.title('dimension %s'%str(i_dim+1))   
    
def plot_t(t,p,x,h=None,color=None,label=None):
    if color is None: color = 'darkorange'
        
    if t.shape[0]>5000:
        rand_idx=np.random.permutation(x.shape[0])[0:5000]
        t = t[rand_idx]
        p = p[rand_idx]
        x = x[rand_idx]
        if h is not None: h = h[rand_idx]
            
    if len(x.shape)==1:
        sort_idx = x.argsort()
        if h is None:
            plt.scatter(x,p,alpha=0.1,color='royalblue')
        else:
            plt.scatter(x[h==0],p[h==0],alpha=0.1,color='royalblue')
            plt.scatter(x[h==1],p[h==1],alpha=0.1,color='seagreen')
        plt.plot(x[sort_idx],t[sort_idx],color=color,label=label)
        plt.ylim([0,2*t.max()])
            
    else:
        n_plot = min(x.shape[1],4)
        for i in range(n_plot):
            plt.subplot(str(n_plot)+'1'+str(i+1))
            sort_idx=x[:,i].argsort()
            if h is None:
                plt.scatter(x[:,i],p,alpha=0.1)
            else:
                plt.scatter(x[:,i][h==0],p[h==0],alpha=0.1,color='royalblue')
                plt.scatter(x[:,i][h==1],p[h==1],alpha=0.1,color='seagreen')
            plt.scatter(x[:,i][sort_idx],t[sort_idx],s=8,alpha=0.2,color='darkorange')
            plt.ylim([0,2*t.max()])
    #plt.scatter(x,t,alpha=0.2)
    #plt.ylim([0,1.2*t.max()])
    #plt.ylabel('t')
    #plt.xlabel('x')
    
def plot_scatter_t(t,p,x,h=None,color='orange',label=None):        
    if t.shape[0]>5000:
        rand_idx=np.random.permutation(x.shape[0])[0:5000]
        t = t[rand_idx]
        p = p[rand_idx]
        x = x[rand_idx]
        if h is not None: 
            h = h[rand_idx]            
    sort_idx = x.argsort()
    if h is None:
        plt.scatter(x,p,alpha=0.1,color='steelblue')
    else:
        plt.scatter(x[h==0],p[h==0],alpha=0.1,color='steelblue')
        plt.scatter(x[h==1],p[h==1],alpha=0.3,color='seagreen')
    plt.scatter(x[sort_idx],t[sort_idx],color=color,s=4,alpha=0.6,label=label)
    plt.ylim([0, 1.5*t.max()])
        
def plot_data_1d(p,x,h,n_pt=1000):
    rnd_idx=np.random.permutation(p.shape[0])[0:n_pt]
    p = p[rnd_idx]
    x = x[rnd_idx]
    h = h[rnd_idx]
    plt.scatter(x[h==1],p[h==1],color='r',alpha=0.2,label='alt')
    plt.scatter(x[h==0],p[h==0],color='b',alpha=0.2,label='null')
    plt.xlabel('covariate')
    plt.ylabel('p-value')
    plt.title('hypotheses') 
    
def plot_data_2d(p,x,h,n_pt=1000):
    rnd_idx=np.random.permutation(p.shape[0])[0:n_pt]
    p = p[rnd_idx]
    x = x[rnd_idx,:]
    h = h[rnd_idx]
    plt.scatter(x[h==1,0],x[h==1,1],color='r',alpha=0.2,label='alt')
    plt.scatter(x[h==0,0],x[h==0,1],color='b',alpha=0.2,label='null')
    plt.xlabel('covariate 1')
    plt.ylabel('covariate 2')
    plt.title('hypotheses') 

"""
    ancillary functions
""" 
def sigmoid(x):
    x = x.clip(min=-20,max=20)
    return 1/(1+np.exp(-x))
        

def inv_sigmoid(w):
    w = w.clip(min-1e-8,max=1-1e-8)
    return np.log(w/(1-w))