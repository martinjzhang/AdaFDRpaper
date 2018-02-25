import numpy as np 
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from util import *
from matplotlib import mlab


## generating the 1d toy example
def toy_data_1d(job_id=0,n_sample=10000,vis=0):
    def pi1_gen(x): # need to be fixed 
        pi1=0.03*sp.stats.norm.pdf(x,loc=0.2,scale=0.05)+0.04*sp.stats.norm.pdf(x,loc=0.8,scale=0.05)
        pi1+=0.15*x 
        return pi1

    def plot_pi1_1d(pi1_gen):
        x_grid   = np.linspace(0,1,100)
        pi1_grid = pi1_gen(x_grid)  
        plt.plot(x_grid,pi1_grid)
        plt.xlabel('covariate')
        plt.ylabel('alt distribution')
        plt.title('the alternative distribution')
    
    np.random.seed(42)
    if job_id == 0: # Gaussian mixtures     
        x   = np.random.uniform(0,1,size=n_sample)
        pi1 = pi1_gen(x)
        p   = np.zeros(n_sample)
        
        # generating the hypothesis       
        h  = np.array((np.random.uniform(size=n_sample)<pi1),dtype=int)
        n0 = np.sum(h==0)
        n1 = np.sum(h==1)
        
        # generating the p-values 
        p[h==0] = np.random.uniform(size=n0)
        p[h==1] = np.random.beta(a=0.4,b=4,size=n1)
        
        #plt.figure()
        #plt.hist(p[h==1],bins=100)
        #plt.show()
        #print(np.mean(p[h==1]))
        
        if vis == 1:
            print('### Summary ###')
            print('# null: %s, # alt: %s:, null proportion: %s'%(str(np.sum(h==0)),str(np.sum(h==1)),str(np.sum(h==0)/h.shape[0])))
            plt.figure(figsize=[16,5])
            plt.subplot(121)
            plot_pi1_1d(pi1_gen)
            plt.subplot(122)
            plot_data_1d(p,x,h)
            plt.legend()
            plt.show() 
        return p,x,h
    
## neuralFDR simulated examples    
def neuralfdr_generate_data_1D(job=0, n_samples=10000,data_vis=0, num_case=4):
    if job == 0: # discrete case
        pi1=np.random.uniform(0,0.3,size=num_case)
        X=np.random.randint(0, num_case, n_samples)
        
        p = np.zeros(n_samples)
        h = np.zeros(n_samples)
        
        for i in range(n_samples):
            rnd = np.random.uniform()
            if rnd > pi1[X[i]]:
                p[i] = np.random.uniform()
                h[i] = 0
            else:
                p[i] = np.random.beta(a = np.random.uniform(0.2,0.4), b = 4)
                h[i] = 1
        return p,h,X
    
def neuralfdr_generate_data_1D_cont(pi1, X, job=0):
    if job == 0: # discrete case
        n_samples = len(X)
        p = np.zeros(n_samples)
        h = np.zeros(n_samples)
        
        for i in range(n_samples):
            rnd = np.random.uniform()
            if rnd > pi1[i]:
                p[i] = np.random.uniform()
                h[i] = 0
            else:
                p[i] = np.random.beta(a = np.random.uniform(0.2,0.4), b = 4)
                h[i] = 1
        return p, h, X
    
def neuralfdr_generate_data_2D(job=0, n_samples=100000,data_vis=0):
    np.random.seed(42)
    if job == 0: # Gaussian mixtures 
        x1 = np.random.uniform(-1,1,size = n_samples)
        x2 = np.random.uniform(-1,1,size = n_samples)
        pi1 = ((mlab.bivariate_normal(x1, x2, 0.25, 0.25, -0.5, -0.2)+
               mlab.bivariate_normal(x1, x2, 0.25, 0.25, 0.7, 0.5))/2).clip(max=1)        
        p = np.zeros(n_samples)
        h = np.zeros(n_samples)
               
        for i in range(n_samples):
            rnd = np.random.uniform()
            if rnd > pi1[i]:
                p[i] = np.random.uniform()
                h[i] = 0
            else:
                p[i] = np.random.beta(a = 0.3, b = 4)
                h[i] = 1
        X = np.concatenate([[x1],[x2]]).T
        X = (X+1)/2
      
        if data_vis == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            x_grid = np.arange(-1, 1, 1/100.0)
            y_grid = np.arange(-1, 1, 1/100.0)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            pi1_grid = ((mlab.bivariate_normal(X_grid, Y_grid, 0.25, 0.25, -0.5, -0.2)+
               mlab.bivariate_normal(X_grid, Y_grid, 0.25, 0.25, 0.7, 0.5))/2).clip(max=1)  
            ax1.pcolor(X_grid, Y_grid, pi1_grid)
            
            ax2 = fig.add_subplot(122)
            alt=ax2.scatter(x1[h==1][1:50], x2[h==1][1:50],color='r')
            nul=ax2.scatter(x1[h==0][1:50], x2[h==0][1:50],color='b')
            ax2.legend((alt,nul),('50 alternatives', '50 nulls'))
            
        return p, h, X
    if job == 1: # Linear trend
        
        x1 = np.random.uniform(-1,1,size = n_samples)
        x2 = np.random.uniform(-1,1,size = n_samples)
        pi1 = 0.1 * (x1 + 1) /2 +  0.3 *(1-x2) / 2
        
        p = np.zeros(n_samples)
        h = np.zeros(n_samples)
         
        for i in range(n_samples):
            rnd = np.random.uniform()
            if rnd > pi1[i]:
                p[i] = np.random.uniform()
                h[i] = 0
            else:
                p[i] = np.random.beta(a = 0.3, b = 4)
                h[i] = 1
        X = np.concatenate([[x1],[x2]]).T
        X = (X+1)/2
        
        if data_vis == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            x_grid = np.arange(-1, 1, 1/100.0)
            y_grid = np.arange(-1, 1, 1/100.0)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            pi1_grid =  0.1 * (X_grid + 1) /2 +  0.3 *(1-Y_grid) / 2
            
            ax1.pcolor(X_grid, Y_grid, pi1_grid)
            
            ax2 = fig.add_subplot(122)
            alt=ax2.scatter(x1[h==1][1:50], x2[h==1][1:50],color='r')
            nul=ax2.scatter(x1[h==0][1:50], x2[h==0][1:50],color='b')
            ax2.legend((alt,nul),('50 alternatives', '50 nulls'))
            
        return p, h, X
        
        
        
    if job == 2: # Gaussian mixture + linear trend
        x1 = np.random.uniform(-1,1,size = n_samples)
        x2 = np.random.uniform(-1,1,size = n_samples)
        pi1 = ((mlab.bivariate_normal(x1, x2, 0.25, 0.25, -0.5, -0.2)+
               mlab.bivariate_normal(x1, x2, 0.25, 0.25, 0.7, 0.5))/2).clip(max=1)        
        pi1 = pi1 * 0.5 + 0.5*(0.5 * (x1 + 1) /2 +  0.3 *(1-x2) / 2)
        
        p = np.zeros(n_samples)
        h = np.zeros(n_samples)
               
        for i in range(n_samples):
            rnd = np.random.uniform()
            if rnd > pi1[i]:
                p[i] = np.random.uniform()
                h[i] = 0
            else:
                p[i] = np.random.beta(a = 0.3, b = 4)
                h[i] = 1
        X = np.concatenate([[x1],[x2]]).T
        X = (X+1)/2
        
        if data_vis == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            x_grid = np.arange(-1, 1, 1/100.0)
            y_grid = np.arange(-1, 1, 1/100.0)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            pi1_grid = ((mlab.bivariate_normal(X_grid, Y_grid, 0.25, 0.25, -0.5, -0.2)+
               mlab.bivariate_normal(X_grid, Y_grid, 0.25, 0.25, 0.7, 0.5))/2).clip(max=1)  * 0.5 + (0.5 * (0.5 * (X_grid + 1) /2 +  0.3 *(1-Y_grid) / 2))
            ax1.pcolor(X_grid, Y_grid, pi1_grid)
            
            ax2 = fig.add_subplot(122)
            alt=ax2.scatter(x1[h==1][1:50], x2[h==1][1:50],color='r')
            nul=ax2.scatter(x1[h==0][1:50], x2[h==0][1:50],color='b')
            ax2.legend((alt,nul),('50 alternatives', '50 nulls'))
            
        return p, h, X
    
def load_airway(verbose=False):
    file_name='/data/martin/NeuralFDR/NeuralFDR_data/data_airway.csv'
    X = np.loadtxt(file_name,skiprows=1,delimiter=',')
    x=X[:,0]
    p=X[:,1]    
    if verbose:
        print('## airway data ##')
        print('# hypothesis: %s'%str(x.shape[0]))
        for i in range(5):
            print('p=%s, x=%s'%(str(p[i]),str(x[i])))
        print('\n')
    return p,x

def load_proteomics(verbose=False):
    file_name='/data/martin/NeuralFDR/NeuralFDR_data/proteomics.csv'
    X = np.loadtxt(file_name,skiprows=1,delimiter=',')
    x=X[:,0]
    p=X[:,1]    
    if verbose:
        print('## proteomics.csv ##')
        print('# hypothesis: %s'%str(x.shape[0]))
        for i in range(5):
            print('p=%s, x=%s'%(str(p[i]),str(x[i])))
        print('\n')
    return p,x

def load_GTEx_1d(verbose=False):
    file_name='/data/martin/NeuralFDR/NeuralFDR_data/data_gtex.csv'
    X = np.loadtxt(file_name,skiprows=1,delimiter=',')
    x=X[:,0]
    p=X[:,1]    
    if verbose:
        print('## airway data ##')
        print('# hypothesis: %s'%str(x.shape[0]))
        for i in range(5):
            print('p=%s, x=%s'%(str(p[i]),str(x[i])))
        print('\n')
    return p,x

def load_GTEx_full(verbose=False):
    file_name='/data/martin/NeuralFDR/NeuralFDR_data/gtex_full.csv'
    X = np.loadtxt(file_name,skiprows=1,delimiter=',')
    x=X[:,0:3]
    p=X[:,3]    
    if verbose:
        print('## airway data ##')
        print('# hypothesis: %s'%str(x.shape[0]))
        for i in range(5):
            print('p=%s, x=%s'%(str(p[i]),str(x[i])))
        print('\n')
    return p,x