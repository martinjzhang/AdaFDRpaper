import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

""" 
    preprocessing: standardize the hypothesis features 
""" 

def x_prep(x_,verbose=False):
    x = x_.copy()
    d=1 if len(x.shape)==1 else x.shape[1]
    if verbose:
        print('Pre prep')
        plt.figure(figsize=[18,5])
        for i in range(min(x.shape[1],3)):
            plt.subplot('13'+str(i+1))
            plt.hist(x[:,i],bins=50)
            plt.title('dimension %s'%str(i+1))
        plt.show()                
    
    ## preprocesing    
    x = rank(x)
    
    if d==1:    
        x = (x-x.min())/(x.max()-x.min()) 
    else: 
        for i in range(x.shape[1]):
            x[:,i] = (x[:,i]-x[:,i].min())/(x[:,i].max()-x[:,i].min())     
    
    if verbose:
        print('Post prep')
        plt.figure(figsize=[18,5])
        for i in range(min(x.shape[1],3)):
            plt.subplot('13'+str(i+1))
            plt.hist(x[:,i],bins=50)
            plt.title('dimension %s'%str(i+1))
        plt.show()
    return x

## baseline comparison methods
def bh(p,alpha=0.05,n_sample=None,verbose=False):
    if n_sample is None: n_sample = p.shape[0]
    p_sort   = sorted(p)
    n_rej    = 0
    for i in range(p.shape[0]):
        if p_sort[i] < i*alpha/n_sample:
            n_rej = i
    t_rej = p_sort[n_rej]
    if verbose:
        print("### bh summary ###")
        print("# rejections: %s"%str(n_rej))
        print("rejection threshold: %s"%str(t_rej))
        print("\n")
    return n_rej,t_rej

def storey_bh(p,alpha=0.05,lamb=0.6,n_sample=None,verbose=False):
    if n_sample is None: n_sample = p.shape[0]
    pi0_hat  = np.sum(p>lamb)/n_sample/(1-lamb)
    alpha   /= pi0_hat
    p_sort = sorted(p)
    n_rej = 0
    for i in range(p.shape[0]):
        if p_sort[i] < i*alpha/n_sample:
            n_rej = i
    t_rej = p_sort[n_rej]
    if verbose:
        print("### sbh summary ###")
        print("# rejections: %s"%str(n_rej))
        print("rejection threshold: %s"%str(t_rej))
        print("null estimate: %s"%str(pi0_hat))
        print("\n")
    return n_rej,t_rej,pi0_hat

# summary functions 
def evaluation(p,t,h):
    print("### Testing Summary ###")
    print("# rejections: %s"%str(np.sum(p<t)))
    print('FDP: %s'%str( np.sum((h==0)*(p<t)/np.sum(p<t))))
    print("### End Summary ###\n")
    
def result_summary(pred,h):
    print("## Testing Summary ##")
    if h is not None: print("Num of alternatives:",np.sum(h))
    print("Num of discovery:",np.sum(pred))
    if h is not None: print("Num of true discovery:",np.sum(pred * h))
    if h is not None: print("Actual FDP:", 1-np.sum(pred * h) / np.sum(pred))
    print('\n')

# ancillary functions for PrimFDR
def rank(x):
    ranks = np.empty_like(x)
    if len(x.shape)==1:
        temp = x.argsort(axis=0)       
        ranks[temp] = np.arange(x.shape[0])
    else:
        for i in range(x.shape[1]):
            temp = x[:,i].argsort(axis=0)       
            ranks[temp,i] = np.arange(x.shape[0])
    return ranks
    
    
    
def plot_t(t,x):
    plt.scatter(x,t,alpha=0.2)
    plt.ylim([0,1.2*t.max()])
    plt.ylabel('t')
    plt.xlabel('x')
    
    
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

    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def inv_sigmoid(w):
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