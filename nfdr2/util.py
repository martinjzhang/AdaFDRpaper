import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import logging

"""
    calculate the dimension-wise rank statistics
    # fix it: for discrete features, it may be nice to keep their values the same
    
    ----- input  -----
    x: an n*d array 
    
    ----- output -----
    ranks: an n*d array, column-wise rank of x
"""
def rank(x,continue_rank=True):
    ranks = np.empty_like(x)
    n,d = x.shape
    for i in range(d):
        if continue_rank:           
            temp = x[:,i].argsort(axis=0)       
            ranks[temp,i] = np.arange(n)
        else:
            ranks[:,i] = rankdata(x[:,i])
    return ranks

"""
    rescale to have the mirror estimate below level alpha
"""
def rescale_mirror(t,p,alpha,f_write=None,title=''):
    if f_write is not None:
        f_write.write('\n## rescale_mirror: %s\n'%title)
        #f_write.write('# quantile of t (1,25,75,99): %s\n'%(np.percentile(t,[1,25,75,99])))
    
    # first rescale t to a sensible region, accounts for different range of t
    t_999 = np.percentile(t,99.9)
    if t.clip(max=t_999).mean()>0.2:
        gamma1 = 0.2/t.clip(max=t_999).mean()
    else: 
        gamma1=1
    t = t*gamma1
    if f_write is not None:
        f_write.write('# quantile of t (1,25,75,99): %s\n'%(np.percentile(t,[1,25,75,99])))
        f_write.write('# gamma1=%0.4f\n'%gamma1)
    
    # binary search
    gamma_l = 0
    gamma_u = 0.2/np.mean(t)
    gamma_m = (gamma_u+gamma_l)/2
    
    while gamma_u-gamma_l>1e-4 or (np.sum(p>1-t*gamma_m)/np.sum(p<t*gamma_m) > alpha):
        gamma_m = (gamma_l+gamma_u)/2
        D_hat = np.sum(p<t*gamma_m)
        FD_hat = np.sum(p>1-t*gamma_m)
        alpha_hat = FD_hat/D_hat
         
        if f_write is not None:
            f_write.write('# gamma_l=%0.4f, gamma_u=%0.4f, D_hat=%d, FD_hat=%d, alpha_hat=%0.4f\n'%\
                         (gamma_l,gamma_u,D_hat,FD_hat,alpha_hat))
        if alpha_hat < alpha:
            gamma_l = gamma_m
        else: 
            gamma_u = gamma_m
    
    if f_write is not None:
        f_write.write('# final output: gamma=%0.4f\n'%((gamma_u+gamma_l)/2*gamma1))
    return (gamma_u+gamma_l)/2*gamma1 

#"""
#    rescale to have the mean-t estimate below level alpha 
#    this is not used here 
#"""
#
#def rescale_naive(t,p,alpha,nr=1,verbose=False):
#    print('### rescale_naive ###')
#    gamma_l = 0
#    gamma_u = 1/np.mean(t)
#    gamma_m = (gamma_l+gamma_u)/2
#    while gamma_u-gamma_l>1e-8 or (np.sum(t*gamma_m)*nr/np.sum(p<t*gamma_m)>alpha):
#        print(gamma_l,gamma_u,np.sum(t*gamma_m)*nr,np.sum(p<t*gamma_m))
#        gamma_m = (gamma_l+gamma_u)/2
#        if np.sum(t*gamma_m)*nr/np.sum(p<t*gamma_m) < alpha:
#            gamma_l = gamma_m
#        else: 
#            gamma_u = gamma_m
#    print('\n')
#    return (gamma_u+gamma_l)/2
        
"""
    calculate the threshold based on the learned mixture: trend+bump
"""
def t_cal(x,a,b,w,mu,sigma):
    K = w.shape[0]
    if len(x.shape)==1:
        t = np.exp(x*a+b)
    else:
        t = np.exp(x.dot(a)+b)
    for i in range(K):
        if len(x.shape)==1:
            #t += np.exp(w[i])*np.exp(-(x-mu[i])**2/sigma[i])
            t += np.exp(w[i])*np.exp(-(x-mu[i])**2*sigma[i])
        else:
            #t += np.exp(w[i])*np.exp(-np.sum((x-mu[i])**2/sigma[i],axis=1))
            t += np.exp(w[i])*np.exp(-np.sum((x-mu[i])**2*sigma[i],axis=1))
    return t.clip(max=1)
    
""" 
    summary the result based on the predicted value and the true value 
""" 
def result_summary(pred,h,logger=None,f_write=None):
    
    if h is not None: 
        print("# Num of alternatives:",np.sum(h))
    print("# Num of discovery:",np.sum(pred))
    if logger is not None:
        logger.info('# n_rej=%d'%np.sum(pred))
    if f_write is not None:
        f_write.write('# n_rej=%d\n'%np.sum(pred))
    if h is not None: 
        print("# Num of true discovery: %d"%np.sum(pred*h))
        print("# Actual FDP: %0.3f"%(1-np.sum(pred * h) / np.sum(pred)))
        if f_write is not None:
            f_write.write("# Num of true discovery: %d\n"%np.sum(pred*h))
            f_write.write("# Actual FDP: %0.3f\n"%(1-np.sum(pred * h) / np.sum(pred)))     
    print('')
    if f_write is not None:
        f_write.write('\n')
    return

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
                plt.scatter(x[:,i][h==1],p[h==1],alpha=0.1,color='seegreen')
            plt.scatter(x[:,i][sort_idx],t[sort_idx],s=8,alpha=0.2,color='darkorange')
            plt.ylim([0,2*t.max()])
    #plt.scatter(x,t,alpha=0.2)
    #plt.ylim([0,1.2*t.max()])
    #plt.ylabel('t')
    #plt.xlabel('x')
    
def plot_scatter_t(t,p,x,h=None,color=None,label=None):
    if color is None: color = 'darkorange'
        
    if t.shape[0]>5000:
        rand_idx=np.random.permutation(x.shape[0])[0:5000]
        t = t[rand_idx]
        p = p[rand_idx]
        x = x[rand_idx]
        if h is not None: h = h[rand_idx]
            
    sort_idx = x.argsort()
    if h is None:
        plt.scatter(x,p,alpha=0.1,color='royalblue')
    else:
        plt.scatter(x[h==0],p[h==0],alpha=0.1,color='royalblue')
        plt.scatter(x[h==1],p[h==1],alpha=0.3,color='seagreen')
    plt.scatter(x[sort_idx],t[sort_idx],color=color,s=4,alpha=0.6,label=label)
    plt.ylim([0,2*t.max()])
        
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
      
    
''' A simple profiler for logging '''
import logging
import time

class Profiler(object):
    def __init__(self, name, logger=None, level=logging.INFO, enable=True):
        self.name = name
        self.logger = logger
        self.level = level
        self.enable = enable

    def step( self, name ):
        """ Returns the duration and stepname since last step/start """
        duration = self.summarize_step( start=self.step_start, step_name=name, level=self.level )
        now = time.time()
        self.step_start = now
        return duration 

    def __enter__( self ):
        self.start = time.time()
        self.step_start = time.time()
        return self
 
    def __exit__( self, exception_type, exception_value, traceback ):
        if self.enable:
            self.summarize_step( self.start, step_name="complete" )
        
        

    def summarize_step( self, start, step_name="", level=None ):
        duration = time.time() - start
        step_semicolon = ':' if step_name else ""
        #if self.logger:
        #    level = level or self.level
            #self.logger.log( self.level, "{name}{step}: ".format( name=self.name, step=step_semicolon + " " + step_name, secs=1/duration) )
        #else:
        print("{name}{step}:  {duration:.5f} seconds".format( name=self.name, step=step_semicolon + " " + step_name, duration=duration))
        return duration