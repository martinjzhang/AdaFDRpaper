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

def load_2DGM(n_samples=100000,verbose=False):
    np.random.seed(42)
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
      
    if verbose:
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
    return p,h,X

def load_2Dslope(n_samples=100000,verbose=False):
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
    
    if verbose:
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
    return p,h,X

def load_2DGM_slope(n_samples=100000,verbose=False):
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
    
    if verbose:
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
    return p,h,X

def load_5DGM(n_sample=100000,verbose=False):
    p,h,x = load_2DGM(n_samples=n_sample,verbose=verbose) 
    x_noise = np.random.uniform(high=1,low=-1,size = (n_sample,3))
    x = np.concatenate([x,x_noise],1)
    return p,h,x

def load_100D(n_sample=100000,verbose=False):
    def generate_data_1D_cont(pi1,X):
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
    
    X = np.random.uniform(high = 5, size = (n_sample,))
    pi1 = (5-X) / 10.0
    p, h, x = generate_data_1D_cont(pi1, X)
    x_noise = np.random.uniform(high = 5, size = (n_sample,99))
    x = np.concatenate([np.expand_dims(x,1), x_noise], 1)
    return p,h,x


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

"""
    load the GTEx full data 
    data are only kept for those with p-values >0.995 or <0.005
    the full data size is 10623893
""" 
def load_GTEx_full(verbose=False):
    file_name='/data/martin/NeuralFDR/NeuralFDR_data/gtex_new_filtered.csv'
    X = np.loadtxt(file_name,skiprows=1,delimiter=',')
    x,p,n_full = X[:,0:4],X[:,4],10623893
    x[:,0],x[:,1] = np.log(x[:,0]+1), np.log(x[:,1]+1)
    if verbose:
        print('## Load GTEx full data ##')
        print('# all hypothesis: %d'%n_full)
        print('# filtered hypothesis: %d'%x.shape[0])
        for i in range(5):
            print('# p=%s, x=%s'%(str(p[i]),str(x[i])))
        print('\n')
        
    cate_name = {'Art': 0, 'Ctcf': 1, 'CtcfO': 2, 'DnaseD': 3, 'DnaseU': 4, 'Elon': 5, 'ElonW': 6, 'Enh': 7, 'EnhF': 8, 'EnhW': 9, 'EnhWF': 10, 'FaireW': 11, "Gen3'": 12, "Gen5'": 13, 'H4K20': 14, 'Low': 15, 'Pol2': 16, 'PromF': 17, 'PromP': 18, 'Quies': 19, 'Repr': 20, 'ReprD': 21, 'ReprW': 22, 'Tss': 23, 'TssF': 24}
    cate_name = {v: k for k, v in cate_name.items()}
    cate_name = [None,None,None,cate_name]

    return p,x,n_full,cate_name

def load_ukbb(verbose=False):
    file_name='/data/martin/ukbb.csv'
    X = np.loadtxt(file_name,skiprows=1,delimiter=',')
    x,p,n_full = X[:,0:2],X[:,2],847800
    #x[:,0],x[:,1] = np.log(x[:,0]+1), np.log(x[:,1]+1)
    if verbose:
        print('## Load GTEx full data ##')
        print('# all hypothesis: %d'%n_full)
        print('# filtered hypothesis: %d'%x.shape[0])
        for i in range(5):
            print('# p=%s, x=%s'%(str(p[i]),str(x[i])))
        print('\n')
        
    cate_name = {'Art': 0, 'Ctcf': 1, 'CtcfO': 2, 'DnaseD': 3, 'DnaseU': 4, 'Elon': 5, 'ElonW': 6, 'Enh': 7, 'EnhF': 8, 'EnhW': 9, 'EnhWF': 10, 'FaireW': 11, "Gen3'": 12, "Gen5'": 13, 'H4K20': 14, 'Low': 15, 'Pol2': 16, 'PromF': 17, 'PromP': 18, 'Quies': 19, 'Repr': 20, 'ReprD': 21, 'ReprW': 22, 'Tss': 23, 'TssF': 24}
    cate_name = {v: k for k, v in cate_name.items()}
    cate_name = [cate_name,None]

    return p,x,n_full,cate_name