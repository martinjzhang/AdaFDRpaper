import numpy as np
import scipy as sp
from scipy import stats
from sklearn.mixture import GaussianMixture
import torch
from torch.autograd import Variable
from util import *
from multiprocessing import Pool
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as tf
#from torch.multiprocessing import Pool

""" 
    preprocessing: standardize the hypothesis features 
    
    ----- input  -----
    x_: np_array, the hypothesis features.
    qt_norm: bool, if perform quantile normalization.
    return_metainfo: bool, if return the meta information regarding the features
    vis_dim: list, the dimensions to visualize. counting starts from 0, needed only when verbose is True
    verbose: bool, if generate ancillary information
    
    ----- output -----
    
""" 

def feature_preprocess(x_,p,qt_norm=True,continue_rank=True,return_metainfo=False,vis_dim=None,verbose=False):
    def cal_meta_info(x,p):
        if np.unique(x).shape[0]<100:
            x_new,x_order_new = reorder_discrete_feature(x,p) 
            return x_new,['discrete',x_order_new]
        else: 
            return x,['continuous',None]  
        
    def reorder_discrete_feature(x,p):
        ## separate the null and the alt proportion
        _,t_BH = bh(p,alpha=0.1)
        x_null,x_alt = x[p>0.5],x[p<t_BH]  
        x_val = np.unique(x)

        ## calculate the ratio
        cts_null = np.zeros([x_val.shape[0]],dtype=int)
        cts_alt  = np.zeros([x_val.shape[0]],dtype=int)
        for i,val in enumerate(x_val):
            cts_null[i],cts_alt[i] = (x_null==val).sum(),(x_alt==val).sum()
        p_null  = (cts_null+1)/np.sum(cts_null+1)
        p_alt   = (cts_alt+1)/np.sum(cts_alt+1)      
        p_ratio = p_alt/p_null 
            
        ## resort according to the ratio
        idx_sort    = p_ratio.argsort()
        x_new       = np.copy(x)
        x_order_new = []
        for i in range(x_val.shape[0]):
            x_new[x==x_val[idx_sort[i]]] = x_val[i]   
            x_order_new.append(idx_sort[i])
        return x_new,x_order_new
    
    x = x_.copy()
    d=1 if len(x.shape)==1 else x.shape[1]   
        
    ## feature visualization before preprocessing
    if verbose:
        plt.figure(figsize=[18,5])
        plt.suptitle('before feature_preprocess')
        plot_x(x,vis_dim=vis_dim)       
        plt.show()                
    
    ## preprocesing
    # reorder the discrete features as well as calculating the meta information
    meta_info = []
    if d==1:
        x,temp_meta = cal_meta_info(x,p)
        meta_info.append(temp_meta)
    else:
        for i in range(d):
            x[:,i],temp_meta = cal_meta_info(x[:,i],p)
            meta_info.append(temp_meta)
        
    # quantile normalization
    if qt_norm: x=rank(x)
        
    # scale to be between 0 and 1
    x = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)) 
    
    if verbose:
        plt.figure(figsize=[18,5])
        plt.suptitle('after feature_preprocess')
        plot_x(x,vis_dim=vis_dim) 
        plt.show()
    
    if return_metainfo:
        return x,meta_info
    else:
        return x
    
""" 
    feature_explore: provide a visualization of pi1/pi0 for each dimension, to visualize the amount of information carried by each dimension
    ----- input  -----
    
    ----- output -----
    
"""
def feature_explore(p,x,alpha=0.1,qt_norm=False,vis_dim=None,cate_name={}):
    def plot_feature_1d(x_null,x_alt,bins,meta_info,title='',cate_name=None):
        feature_type,cate_order = meta_info        
        if feature_type == 'continuous':         
            ## continuous feature: using kde to estimate 
            n_bin = bins.shape[0]-1
            x_grid = (bins+(bins[1]-bins[0])/2)[0:-1]
            p_null,_ = np.histogram(x_null,bins=bins) 
            p_alt,_= np.histogram(x_alt,bins=bins)         
            p_null = (p_null+1)/np.sum(p_null+1)*n_bin
            p_alt = (p_alt+1)/np.sum(p_alt+1)*n_bin
            kde_null = stats.gaussian_kde(x_null).evaluate(x_grid)
            kde_alt = stats.gaussian_kde(x_alt).evaluate(x_grid)
            p_ratio = (kde_alt+1e-2)/(kde_null+1e-2)        
                 
        else: 
            ## discrete feature: directly use the empirical counts 
            unique_null,cts_null = np.unique(x_null,return_counts=True)
            unique_alt,cts_alt = np.unique(x_alt,return_counts=True)            
            unique_val = np.array(list(set(list(unique_null)+list(unique_alt))))
            unique_val = np.sort(unique_val)            
            p_null,p_alt = np.zeros([unique_val.shape[0]]),np.zeros([unique_val.shape[0]])          
            for i,key in enumerate(unique_null): p_null[unique_val==key] = cts_null[i]                
            for i,key in enumerate(unique_alt): p_alt[unique_val==key] = cts_alt[i]           
            n_bin = unique_val.shape[0]           
            p_null = (p_null+1)/np.sum(p_null+1)*n_bin
            p_alt = (p_alt+1)/np.sum(p_alt+1)*n_bin            
            p_ratio = (p_alt+1e-2)/(p_null+1e-2)  
            x_grid = (np.arange(unique_val.shape[0])+1)/(unique_val.shape[0]+1)
            
            if cate_name is None: 
                cate_name_ = cate_order
            else:
                cate_name_ = []
                for i in cate_order:
                    cate_name_.append(cate_name[i])
                    
        plt.figure(figsize=[18,5])
        plt.subplot(121)
        plt.bar(x_grid,p_null,width=1/n_bin,color='royalblue',alpha=0.6,label='null')
        plt.bar(x_grid,p_alt,width=1/n_bin,color='darkorange',alpha=0.6,label='alt')
        plt.xlim([0,1])
        if feature_type=='discrete': plt.xticks(x_grid,cate_name_,rotation=45)
        plt.title('estimated null/alt proportion')
        plt.legend()
        plt.subplot(122)
        if feature_type == 'continuous':
            plt.plot(x_grid,p_ratio,color='seagreen',label='ratio',linewidth=4) 
        else:
            plt.plot(x_grid,p_ratio,color='seagreen',marker='o',label='ratio',linewidth=4)
            plt.xticks(x_grid,cate_name_,rotation=45)
        plt.xlim([0,1])
        plt.title('ratio')
        plt.suptitle(title+' (%s)'%feature_type)
        plt.show()
    
    
    d=1 if len(x.shape)==1 else x.shape[1]
    ## preprocessing
    x,meta_info = feature_preprocess(x,p,qt_norm=qt_norm,continue_rank=False,return_metainfo=True)   
    
    ## separate the null proportion and the alternative proportion
    _,t_BH = bh(p,alpha=alpha)
    x_null,x_alt = x[p>0.5],x[p<t_BH]      
    
    ## generate the figure
    bins = np.linspace(0,1,51)  
    if vis_dim is None: vis_dim = np.arange(min(4,d))    
    
    if d==1:       
        if 0 in cate_name.keys():
            plot_feature_1d(x_null,x_alt,bins,meta_info[0],title='feature 1',cate_name=cate_name[0])
        else:
            plot_feature_1d(x_null,x_alt,bins,meta_info[0],title='feature 1')
    else: 
        for i in vis_dim:
            if i in cate_name.keys():
                plot_feature_1d(x_null[:,i],x_alt[:,i],bins,meta_info[i],title='feature %s'%str(i+1),cate_name=cate_name[i])
            else:
                plot_feature_1d(x_null[:,i],x_alt[:,i],bins,meta_info[i],title='feature %s'%str(i+1))
       
"""
    PrimFDR: the main testing function 
    fix: add the cross validation wrapper     
"""

def pfdr_test(data):  
    print('pfdr_test start')
    p1,x1 = data[0]
    p2,x2 = data[1]
    p3,x3 = data[2]
    K,alpha,n_itr = data[3]
    print('PrimFDR start')
    _,_,theta = PrimFDR(p1,x1,K=K,alpha=alpha,n_itr=n_itr,verbose=False)
    a,b,w,mu,sigma,gamma = theta
    
    t2 = t_cal(x2,a,b,w,mu,sigma)
    gamma = rescale_mirror(t2,p2,alpha)
    t3 = gamma*t_cal(x3,a,b,w,mu,sigma)
    
    return np.sum(p3<t3),t3,[a,b,w,mu,sigma,gamma]

## fix it: change the wrapper to two-fold cv
def PrimFDR_cv(p,x,K=2,alpha=0.1,n_itr=5000,qt_norm=True,reorder=True,h=None,core='s_core',verbose=False):
    np.random.seed(42)
    color_list = ['navy','orange','darkred']
    x,_ = feature_preprocess(x,p,qt_norm=qt_norm,reorder=reorder)
    n_sample = p.shape[0]
    n_sub = int(n_sample/3)
    d = 1 if len(x.shape)==1 else x.shape[1]
    rand_idx = np.random.permutation(n_sample)
    subidx_list=[rand_idx[0:n_sub],rand_idx[n_sub:2*n_sub],rand_idx[2*n_sub:]]
    
    if verbose:
        start_time=time.time()
        print('#time start: 0.0s')
    
    ## construct the data
    args = [K,alpha,n_itr]
    data = {}
    if d == 1:
        for i in range(3): data[i] = [p[subidx_list[i]],x[subidx_list[i]]]
    else:
        for i in range(3): data[i] = [p[subidx_list[i]],x[subidx_list[i],:]]
            
    Y_input = []    
    Y_input.append([data[1],data[2],data[0],args])
    Y_input.append([data[2],data[0],data[1],args])
    Y_input.append([data[0],data[1],data[2],args])
    if verbose: print('#time input: %0.4fs'%(time.time()-start_time))
    
    if core == 'm_core':
    ## multi-processing
        pool = Pool(3)
        res  = pool.map(pfdr_test, Y_input)
        if verbose: print('#time mp: %0.4fs'%(time.time()-start_time))
        
    ## single core 
    else:
        res=[]
        for i in range(3):
            if verbose: print('## testing fold %d: %0.4fs'%((i+1),time.time()-start_time))
            res.append(pfdr_test(Y_input[i]))
        
    if verbose: print('#time test: %0.4fs'%(time.time()-start_time))
        
    theta = []
    n_rej = []
    t     = []
    
    for i in range(3):
        n_rej.append(res[i][0])
        t.append(res[i][1])
        theta.append(res[i][2])
    
    if verbose:
        print('# total rejection: %d'%np.array(n_rej).sum(), n_rej)
            
        if d==1:
            plt.figure(figsize=[18,5])
            for i in range(3):
                plot_t(t[i],data[i][0],data[i][1],h,color=color_list[i],label='fold '+str(i+1))
                #plot_t(t[i],data[i][0],data[i][1],h,color=color_list[i],label=None)
            plt.legend()
            plt.show()
        print('#time total: %0.4fs'%(time.time()-start_time))     
    
    return n_rej,t,theta    

def PrimFDR(p,x,K=2,alpha=0.1,n_itr=5000,qt_norm=True,h=None,verbose=False,debug='',output_folder=None,logger=None):   
    ## feature preprocessing 
    torch.manual_seed(42)
    d=1 if len(x.shape)==1 else x.shape[1]
    x = feature_preprocess(x,p,qt_norm=qt_norm)
    
    # rough threshold calculation using PrimFDR_init 
    w_init,a_init,mu_init,sigma_init = PrimFDR_init(p,x,K,alpha=alpha,verbose=verbose,output_folder=output_folder,logger=logger) 

    ## transform the parameters
    a     = a_init
    b     = np.log(w_init[0]*(a_init/(np.exp(a_init)-1)).prod())    
    w     = np.log(w_init[1:]/((2*np.pi)**(d/2)*sigma_init.prod(axis=1)))
    mu    = mu_init 
    #sigma = 2*sigma_init**2
    sigma_init = sigma_init.clip(min=1e-3)
    sigma = 1/(2*sigma_init**2)
    
    # scale factor 
    t = t_cal(x,a,b,w,mu,sigma)    
    gamma=rescale_mirror(t,p,alpha)
    t = gamma*t
    b,w = b+np.log(gamma),w+np.log(gamma) 
    
    ## optimization: parameter setting 
    # lambda0: adaptively set based on the approximation accuracy of the sigmoid function
    lambda0,n_rej,n_fd = 1/t.mean(),np.sum(p<t),np.sum(p>1-t)
    while np.absolute(np.sum(sigmoid((lambda0*(t-p))))-n_rej)>0.02*n_rej \
        or np.absolute(np.sum(sigmoid((lambda0*(t+p-1))))-n_fd)>0.02*n_fd:
        lambda0 = lambda0+0.5/t.mean()
    lambda1  = 10/alpha # 100
    lambda0,lambda1 = int(lambda0),int(lambda1)
    loss_rec = np.zeros([n_itr],dtype=float)
    n_samp   = x.shape[0]
    if verbose: 
        print('## optimization paramter:')
        print('# n_itr=%s, n_samp=%s, lambda0=%s, lambda1=%s\n'%(str(n_itr),str(n_samp),str(lambda0),str(lambda1)))
        if logger is not None:
            logger.info('## optimization paramter:')
            logger.info('# n_itr=%s, n_samp=%s, lambda0=%s, lambda1=%s\n'%(str(n_itr),str(n_samp),str(lambda0),str(lambda1)))
           
    ## optimization: initialization         
    lambda0 = Variable(torch.Tensor([lambda0]),requires_grad=False)
    lambda1 = Variable(torch.Tensor([lambda1]),requires_grad=False)
    p       = Variable(torch.from_numpy(p).float(),requires_grad=False)
    x       = Variable(torch.from_numpy(x).float().view(-1,d),requires_grad=False)
    a       = Variable(torch.Tensor([a]),requires_grad=True) if d == 1 else Variable(torch.Tensor(a),requires_grad=True)
    b       = Variable(torch.Tensor([b]),requires_grad=True)
    w       = Variable(torch.Tensor(w),requires_grad=True)
    mu      = Variable(torch.Tensor(mu).view(-1,d),requires_grad=True)
    #sigma = sigma.clip(min=1e-6)
    #sigma_mean = np.mean(1/sigma)
    #print(sigma_mean)
    #sigma   = Variable(torch.Tensor((1/sigma) / (sigma_mean)),requires_grad=True)    
    
    sigma_mean = np.mean(sigma,axis=0)
    #sigma_mean = np.mean(sigma)
    print(sigma_mean)
    sigma   = Variable(torch.Tensor(sigma / sigma_mean),requires_grad=True)
    sigma_mean = Variable(torch.from_numpy(sigma_mean).float(),requires_grad=False)
    
    if verbose:
        print('## optimization initialization:')
        print ('# Slope: a=%s, b=%s'%(str(a.data.numpy()),str(b.data.numpy())))
        for k in range(K):
            print('# Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w.data.numpy()[k]),mu.data.numpy()[k],sigma.data.numpy()[k]))
        print('\n')
        
        if logger is not None:
            logger.info('## optimization initialization:')
            logger.info ('# Slope: a=%s, b=%s'%(str(a.data.numpy()),str(b.data.numpy())))
            for k in range(K):
                logger.info('# Bump %s: w=%s, mu=%s, sigma=%s'\
                            %(str(k),str(w.data.numpy()[k]),mu.data.numpy()[k],sigma.data.numpy()[k]))
            logger.info('\n')
        
    optimizer = torch.optim.Adam([a,b,w,mu,sigma],lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.2)
    optimizer.zero_grad()
    
    ## fix it: tune lambda1 to balance the gradient of the two losses
    for l in range(n_itr):
        scheduler.step()
        ## calculating the model
        optimizer.zero_grad()
        t = torch.exp(torch.matmul(x,a)+b)
        
        for i in range(K):
            t = t+torch.exp(w[i]-torch.matmul((x-mu[i,:])**2,sigma[i,:] * sigma_mean))
        loss1 = -torch.mean(torch.sigmoid(lambda0*(t-p)))
        #loss2 = lambda1*(torch.mean(t)*nr-alpha*torch.mean(torch.sigmoid(lambda0*(t-p)))).clamp(min=0)
        loss2 = lambda1*tf.relu(torch.mean(torch.sigmoid(lambda0*(t+p-1)))-alpha*torch.mean(torch.sigmoid(lambda0*(t-p))))
        loss  = loss1+loss2
        
        ## backprop
        loss.backward()
        optimizer.step()
        
        ## show the result 
        loss_rec[l] = loss.data.numpy()
        
        if verbose:
            if l%(int(n_itr)/5)==0:              
                print('## iteration %s'%str(l))    
                print('n_rej: ',np.sum(t.data.numpy()>p.data.numpy()))
                print('n_rej sig: ',np.sum(sigmoid(lambda0.data.numpy()*(t.data.numpy()-p.data.numpy()))))
                print('FD esti mirror:',np.sum(p.data.numpy()>1-t.data.numpy()))
                print('FD esti mirror sig:',np.sum(sigmoid(lambda0.data.numpy()*(p.data.numpy()-1+t.data.numpy()))))             
                print('loss1: ',loss1.data.numpy())
                print('loss2: ',loss2.data.numpy())
                print('Estimated FDP: %s'\
                      %str((torch.mean(torch.sigmoid(lambda0*(t+p-1)))/torch.mean(torch.sigmoid(lambda0*(t-p)))).data.numpy()))
                print('FDP: %s'%str( np.sum((h==0)*(p.data.numpy()<t.data.numpy()))/np.sum(p.data.numpy()<t.data.numpy())))
                print('Slope: a=%s, b=%s'%(str(a.data.numpy()),str(b.data.numpy())))
                for k in range(K):
                    print('Bump %s: w=%s, mu=%s, sigma=%s'\
                          %(str(k),str(w.data.numpy()[k]),mu.data.numpy()[k],sigma.data.numpy()[k]))             
                                                                      
                if d==1:
                    plt.figure(figsize=[18,5])
                    plot_t(t.data.numpy(),p.data.numpy(),x.data.numpy(),h)
                    if output_folder is not None:
                        plt.savefig(output_folder+'/threshold_itr_%d.png'%l)
                    else:
                        plt.show()
                print('\n')
                
                if logger is not None:
                    logger.info('## iteration %s'%str(l))    
                    logger.info('n_rej: %d'%np.sum(t.data.numpy()>p.data.numpy()))
                    logger.info('n_rej sig: %d'%np.sum(sigmoid(lambda0.data.numpy()*(t.data.numpy()-p.data.numpy()))))
                    logger.info('FD esti mirror: %0.5f'%np.sum(p.data.numpy()>1-t.data.numpy()))
                    logger.info('FD esti mirror sig: %0.5f'\
                                %np.sum(sigmoid(lambda0.data.numpy()*(p.data.numpy()-1+t.data.numpy()))))             
                    logger.info('loss1: %0.5f'%loss1.data.numpy())
                    logger.info('loss2: %0.5f'%loss2.data.numpy())
                    logger.info('Estimated FDP: %0.5f'\
                          %(torch.mean(torch.sigmoid(lambda0*(t+p-1)))/torch.mean(torch.sigmoid(lambda0*(t-p)))).data.numpy())
                    logger.info('FDP: %s'\
                                %str( np.sum((h==0)*(p.data.numpy()<t.data.numpy()))/np.sum(p.data.numpy()<t.data.numpy())))
                    logger.info('Slope: a=%s, b=%s'%(str(a.data.numpy()),str(b.data.numpy())))
                    for k in range(K):
                        logger.info('Bump %s: w=%s, mu=%s, sigma=%s'\
                                    %(str(k),str(w.data.numpy()[k]),mu.data.numpy()[k],sigma.data.numpy()[k]))    
                    logger.info('\n')                  
    if verbose:
        plt.figure()
        plt.plot(np.log(loss_rec-loss_rec.min()+1e-3))
        if output_folder is not None:
            plt.savefig(output_folder+'/loss.png')
        else:
            plt.show()       
        
    p = p.data.numpy()
    x = x.data.numpy()
    
    a,b,w,mu,sigma = a.data.numpy(),b.data.numpy(),w.data.numpy(),mu.data.numpy(),(sigma * sigma_mean).data.numpy()

    t = t_cal(x,a,b,w,mu,sigma)
    gamma = rescale_mirror(t,p,alpha)   
    t *= gamma
    n_rej=np.sum(p<t)     
    if verbose: result_summary(p<t,h)
    theta = [a,b,w,mu,sigma,gamma]
    return n_rej,t,theta

"""
    initialization function of PrimFDR: fit the mixture model with a linear trend and a Gaussian mixture 
"""
def PrimFDR_init(p,x,K,alpha=0.1,n_itr=100,h=None,verbose=False,output_folder=None,logger=None):
    #x = feature_preprocess(x,p)## fix it: multiple places used this function
    np.random.seed(42)
    if verbose: print('## PrimFDR_init starts')   
    if logger is not None: logger.info('## PrimFDR_init starts')   
            
    ## extract the null and the alternative proportion
    _,t_BH = bh(p,alpha=alpha)
    x_null,x_alt = x[p>0.5],x[p<t_BH]
    
    ## fit the null distribution
    if verbose: print('# Learning null distribution')
    if logger is not None: logger.info('# Learning null distribution')   
    w_null,a_null,mu_null,sigma_null = mixture_fit(x_null,K,n_itr=n_itr,\
                                                   verbose=verbose,logger=logger,output_folder=output_folder,suffix='_null')   
    
    x_w = 1/(f_all(x_alt,a_null,mu_null,sigma_null,w_null)+1e-5)
    x_w /= np.mean(x_w)
    
    if verbose: print('# Learning alternative distribution')
    if logger is not None: logger.info('# Learning alternative distribution') 
    w,a,mu,sigma = mixture_fit(x_alt,K,x_w=x_w,n_itr=n_itr,verbose=verbose,\
                               logger=logger,output_folder=output_folder,suffix='_alt')
    
    if verbose:        
        t = f_all(x,a,mu,sigma,w)
        gamma = rescale_mirror(t,p,alpha)   
        t = t*gamma
        print('# Test result with PrimFDR_init')
        if logger is not None: logger.info('# Test result with PrimFDR_init')
        result_summary(p<t,h,logger=logger) 
        
        plt.figure(figsize=[18,5])
        plot_t(t,p,x,h)
        if output_folder is not None: 
            plt.savefig(output_folder+'/Threshold after PrimFDR_init.png')
        else:
            plt.show()
        print('## PrimFDR_init finishes\n')
        if logger is not None: logger.info('## PrimFDR_init finishes\n')
    return w,a,mu,sigma    

""" 
    mixture_fit: fit a GLM+Gaussian mixture using EM algorithm
"""
def mixture_fit(x,K=3,x_w=None,n_itr=1000,verbose=False,debug=False,logger=None,output_folder=None,suffix=None):   
    if len(x.shape)==1: x = x.reshape([-1,1])
    n_samp,d = x.shape
    if x_w is None: x_w=np.ones([n_samp],dtype=float)
        
    ## initialization
    GMM = GaussianMixture(n_components=K,covariance_type='diag').fit(x) # fixit: the weights x_w are gone?? 
    w_old = np.zeros([K+1])
    w = 0.5*np.ones([K+1])/K
    w[0] = 0.5
    a        = ML_slope(x,x_w)   
    mu,sigma = GMM.means_,GMM.covariances_**0.5
    w_samp   = np.zeros([K+1,n_samp],dtype=float)
    i        = 0
    
    if verbose:
        print('## initialization')
        print('Slope: w=%s, a=%s'%(str(w[0]),str(a)))
        for k in range(K):
            print('Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w[k+1]),mu[k],sigma[k]))
        print('\n')    
    
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

    if i >= n_itr and verbose: print('!!! the model does not converge !!!')      
    
    if verbose: 
        print('Slope: w=%s, a=%s'%(str(w[0]),str(a)))
        for k in range(K):
            print('Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w[k+1]),mu[k],sigma[k]))
        print('\n')
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
        
    if output_folder is not None:
        bins_ = np.linspace(0,1,101)
        x_grid = bins_.reshape([-1,1])
        
        if d==1:
            plt.figure(figsize=[18,5])
            plt.hist(x,bins=bins_,weights=x_w/np.sum(x_w)*100)    
            temp_p = np.zeros(bins_.shape[0])
            temp_p += w[0]*f_slope(x_grid,a)
            for i in range(1,w.shape[0]):
                temp_p += w[i]*f_bump(x_grid,mu[i-1],sigma[i-1])
            plt.plot(bins_,temp_p)

            plt.savefig(output_folder+'/projection%s.png'%(suffix))
        
        else:
            plt.figure(figsize=[18,12])
            n_figure = min(d,4)
            for i_dim in range(n_figure):        
                plt.subplot(str(n_figure)+'1'+str(i_dim+1))
                plt.hist(x[:,i_dim],bins=bins_,weights=x_w/np.sum(x_w)*100)    
                temp_p = np.zeros(bins_.shape[0])
                temp_p += w[0]*f_slope(x_grid,a[[i_dim]])
                for i in range(1,w.shape[0]):
                    temp_p += w[i]*f_bump(x_grid,mu[i-1,[i_dim]],sigma[i-1,[i_dim]])
                plt.plot(bins_,temp_p)
                plt.title('Dimension %d'%(i_dim+1))
            plt.savefig(output_folder+'/projection%s.png'%(suffix))
        plt.close('all')
    return w,a,mu,sigma

"""
    sub-routines for mixture_fit
    input:  x: x*d array (not (n,))
            w: (n,) array
            a: (d,) array
            mu,sigma: (k,d) array
            (The constraint on the ndarray is to save the time for type checking)
"""
## ML fit of the slope a/(e^a-1) e^(ax), defined over [0,1]
def ML_slope(x,w=None,c=0.01):   
    a = np.zeros(x.shape[1],dtype=float)
    if w is None:
        w = np.ones(x.shape[0])   
    else:
        w = w+1e-8
    _,d = x.shape
    
    t = np.sum((x.T*w).T,axis=0)/np.sum(w) # sufficient statistic: weighted sum along each dimension
    a_u=100*np.ones([d],dtype=float)
    a_l=-100*np.ones([d],dtype=float)
    ## binary search 
    while np.linalg.norm(a_u-a_l)>0.01:      
        a_m  = (a_u+a_l)/2
        a_m += 1e-2*(a_m==0)
        temp = (np.exp(a_m)/(np.exp(a_m)-1) - 1/a_m +c*a_m)
        a_l[temp<t] = a_m[temp<t]
        a_u[temp>=t] = a_m[temp>=t]
    return (a_u+a_l)/2

# old code
    #def ML_slope_1d(x,w=None,c=0.05):
    #    if w is None:
    #        w = np.ones(x.shape[0])        
    #    t = np.sum(w*x)/np.sum(w) ## sufficient statistic
    #    a_u=100
    #    a_l=-100
    #    ## binary search 
    #    while a_u-a_l>0.1:
    #        a_m  = (a_u+a_l)/2
    #        a_m += 1e-2*(a_m==0)
    #        if (np.exp(a_m)/(np.exp(a_m)-1) - 1/a_m +c*a_m)<t:
    #            a_l = a_m
    #        else: 
    #            a_u = a_m
    #    return (a_u+a_l)/2
    #
    #a = np.zeros(x.shape[1],dtype=float)
    #for i in range(x.shape[1]):
    #    a[i] = ML_slope_1d(x[:,i],w,c=c)
    #return a
    

## slope density function
def f_slope(x,a):
    ## Probability (dimension-wise probability)    
    f_x = np.exp(x*a)
    ## Normalization factor
    norm_factor = np.exp(a)-1  
    temp_v = (norm_factor!=0)
    norm_factor = np.prod(a[temp_v]/norm_factor[temp_v])
    ##  Total probability  
    f_x = np.prod(f_x,axis=1) * norm_factor
    return f_x

# old code:
    #f_x = np.ones([x.shape[0]],dtype=float)     
    #for i in range(x.shape[1]):
    #    if a[i]==0:
    #        f_x *= np.exp(a[i]*x[:,i])
    #    else:
    #        f_x *= a[i]/(np.exp(a[i])-1)*np.exp(a[i]*x[:,i])
            
    #if type(a) is not np.ndarray:
    #    a = np.array([a])
    #if len(x.shape)==1: x = x.reshape([-1,1])
#if len(x.shape)==1: # 1d case 
    #    if a==0:
    #        return f_x
    #    else: 
    #        return a/(np.exp(a)-1)*np.exp(a*x)
    #else:        
    #    for i in range(x.shape[1]):
    #        if a[i]==0:
    #            f_x *= np.exp(a[i]*x[:,i])
    #        else:
    #            f_x *= a[i]/(np.exp(a[i])-1)*np.exp(a[i]*x[:,i])
    #    return f_x

## ML fit of the Gaussian
## Very hard to vectorize. So just leave it here
def ML_bump(x,w=None,logger=None,gradient_check=False):
    def ML_bump_1d(x,w,logger=None,gradient_check=False):
        def fit_f(param,x,w):                
            mu,sigma = param
            Z = sp.stats.norm.cdf(1,loc=mu,scale=sigma)-sp.stats.norm.cdf(0,loc=mu,scale=sigma)
            phi_alpha = 1/np.sqrt(2*np.pi)*np.exp(-mu**2/2/sigma**2)
            phi_beta = 1/np.sqrt(2*np.pi)*np.exp(-(1-mu)**2/2/sigma**2)

            ## average likelihood
            t = np.sum((x-mu)**2*w) / np.sum(w)
            l = -np.log(Z) - np.log(sigma) - t/2/sigma**2        
            ## gradient    
            d_c_mu = 1/sigma * (phi_alpha-phi_beta)
            d_c_sig = 1/sigma * (-mu/sigma*phi_alpha - (1-mu)/sigma*phi_beta)
            d_l_mu = -d_c_mu/Z + np.sum((x-mu)*w)/sigma**2/np.sum(w)
            d_l_sig = -d_c_sig/Z - 1/sigma + t/sigma**3
            grad = np.array([d_l_mu,d_l_sig],dtype=float)  
            return l,grad
        
        ## gradient check
        if gradient_check:
            _,grad_ = fit_f([0.2,0.1],x,w)
            num_dmu = (fit_f([0.2+1e-8,0.1],x,w)[0]-fit_f([0.2,0.1],x,w)[0]) / 1e-8
            num_dsigma = (fit_f([0.2,0.1+1e-8],x,w)[0]-fit_f([0.2,0.1],x,w)[0]) / 1e-8          
            logger.info('## Gradient check ##')
            logger.info('#  param value: mu=%0.3f, sigma=%0.3f'%(0.2,0.1))
            logger.info('#  Theoretical grad: dmu=%0.5f, dsigma=%0.5f'%(grad_[0],grad_[1]))
            logger.info('#  Numerical grad: dmu=%0.5f, dsigma=%0.5f'%(num_dmu,num_dsigma))
            
        ##  if the variance is small, no need to fit a truncated Gaussian
        mu = np.sum(x*w)/np.sum(w)
        sigma = np.sqrt(np.sum((x-mu)**2*w)/np.sum(w))
        
        if sigma<0.1:
            return mu,sigma
        
        param = np.array([mu,sigma])    
        lr = 0.01
        max_step = 0.025
        max_itr = 100
        i_itr = 0
        l_old = -10
        
        #rec_ = []
        while i_itr<max_itr:
            l,grad = fit_f(param,x,w)
            if np.absolute(l-l_old)<0.01:
                break
            else:
                l_old=l
            update = (grad*lr).clip(min=-max_step,max=max_step)
            #rec_.append([i_itr,param,grad,update])
            #print([i_itr,param,grad,update])
            param += update
            i_itr +=1     
            
            if np.isnan(param).any():
                
                return np.mean(x),np.std(x)
                
        mu,sigma = param  
        if sigma>0.25:
            sigma=1
        return mu,sigma      
    
    if w is None: w = np.ones(x.shape[0])
       
    mu    = np.zeros(x.shape[1],dtype=float)
    sigma = np.zeros(x.shape[1],dtype=float)
    for i in range(x.shape[1]):
        mu[i],sigma[i] = ML_bump_1d(x[:,i],w,logger=logger,gradient_check=gradient_check)
    return mu,sigma

#if len(x.shape)==1: x = x.reshape([-1,1])
    #if len(x.shape)==1:
    #    return ML_bump_1d(x,w,logger=logger,gradient_check=gradient_check)
    #else:       
    #    mu    = np.zeros(x.shape[1],dtype=float)
    #    sigma = np.zeros(x.shape[1],dtype=float)
    #    for i in range(x.shape[1]):
    #        mu[i],sigma[i] = ML_bump_1d(x[:,i],w,logger=logger,gradient_check=gradient_check)
    #    return mu,sigma
    
## old code for ML_bump_1d
    #def ML_bump_1d(x,w=None):
    #    if w is None:
    #        w = np.ones(x.shape[0])
    #        
    #    mean_ = np.sum(x*w)/np.sum(w)
    #    var_  = np.sum((x-mean_)**2*w)/np.sum(w)
    #    #print('mean_=%0.3f, var_=%0.3f'%(mean_,var_))
    #    
    #    ## estimation            
    #    mu    = np.sum(x*w)/np.sum(w)    
    #    sigma = np.sqrt(var_)     
    #    return mu,sigma

## bump density function
def f_bump(x,mu,sigma):
    def f_bump_1d(x,mu,sigma): ## correct for the finite interval issue
        if sigma<1e-6: return np.zeros(x.shape)
        pmf = sp.stats.norm.cdf(1,loc=mu,scale=sigma)-sp.stats.norm.cdf(0,loc=mu,scale=sigma)
        return 1/sigma/np.sqrt(2*np.pi)*np.exp(-(x-mu)**2/2/sigma**2)/pmf
       
    f_x = np.ones([x.shape[0]],dtype=float)
    for i in range(x.shape[1]):
        f_x *=  f_bump_1d(x[:,i],mu[i],sigma[i])
    return f_x       
        
# old code  
    #if type(mu) is not np.ndarray:
    #    mu = np.array([mu])   
    #    sigma = np.array([sigma])
    
    #if len(x.shape)==1:
    #    return f_bump_1d(x,mu,sigma)
    #else:
    #    f_x = np.ones([x.shape[0]],dtype=float)
    #    for i in range(x.shape[1]):
    #        f_x *=  f_bump_1d(x[:,i],mu[i],sigma[i])
    #    return f_x       

## the entire density function
def f_all(x,a,mu,sigma,w):
    f = w[0]*f_slope(x,a)        
    for k in range(1,w.shape[0]):
        f += w[k]*f_bump(x,mu[k-1],sigma[k-1])           
    return f

# old code
    #if len(x.shape) == 1:
    #    for k in range(1,w.shape[0]):
    #        f += w[k]*f_bump(x,mu[k-1],sigma[k-1])
    #else:
    #    
    #return f


'''
    baseline comparison methods
'''
def bh(p,alpha=0.1,n_sample=None,verbose=False):
    if n_sample is None: n_sample = p.shape[0]
    p_sort   = sorted(p)
    n_rej    = 0
    for i in range(p.shape[0]):
        if p_sort[i] < i*alpha/n_sample:
            n_rej = i
    t_rej = p_sort[n_rej]
    if verbose:
        print("## bh testing summary ##")
        print("# n_rej = %d"%n_rej)
        print("# t_rej = %0.6f"%t_rej)
        print("\n")
    return n_rej,t_rej

def storey_bh(p,alpha=0.1,lamb=0.5,n_sample=None,verbose=False):
    if n_sample is None: n_sample = p.shape[0]
    pi0_hat  = (np.sum(p>lamb)/(1-lamb)/n_sample).clip(max=1)  
    alpha   /= pi0_hat
    p_sort = sorted(p)
    n_rej = 0
    for i in range(p.shape[0]):
        if p_sort[i] < i*alpha/n_sample:
            n_rej = i
    t_rej = p_sort[n_rej]
    if verbose:
        print("## sbh summary ##")
        print("# n_rej = %d"%n_rej)
        print("# t_rej = %0.6f"%t_rej)
        print("# pi_0 estimate = %0.3f"%pi0_hat)
        print("\n")
    return n_rej,t_rej,pi0_hat


"""
    some old code
""" 
#def PrimFDR_v2(p,x,K,alpha=0.1,n_itr=100,h=None,verbose=False):
#    np.random.seed(42)
#    x = feature_preprocess(x)
#    if verbose:
#        print('## ML initialization starts ##')   
#    
#    ## extract the null and the alternative proportion
#    beta   = -ML_slope(p)
#    _,t_BH = bh(p,alpha=alpha)
#    x_alt  = x[p<t_BH]
#    x_null = x[p>0.5]
#    
#    ## fit the alternative distribution
#    w_null,a_null,mu_null,sigma_null = mixture_fit(x_null,K,verbose=verbose)   
#    w_alt, a_alt, mu_alt, sigma_alt  = mixture_fit(x_alt,K,verbose=verbose) 
#    if verbose:
#        print('## Learned parameters for null population ##')
#        print('Slope: w=%s, a=%s'%(str(w_null[0]),str(a_null)))
#        for k in range(K):
#            print('Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w_null[k+1]),mu_null[k],sigma_null[k]))
#        print('\n')
#        
#    
#    if verbose:
#        print('## Learned parameters for alternative population ##')
#        print('Slope: w=%s, a=%s'%(str(w_alt[0]),str(a_alt)))
#        for k in range(K):
#            print('Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w_alt[k+1]),mu_alt[k],sigma_alt[k]))
#        print('\n')
#    
#    if verbose:
#        #print('## Learned parameters ##')
#        #print('Slope: w=%s, a=%s'%(str(w[0]),str(a)))
#        #for k in range(K):
#        #    print('Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w[k+1]),mu[k],sigma[k]))
#        #print('\n')
#        
#        pi_null = w_null[0]*f_slope(x,a_null)
#        for k in range(K):
#            pi_null += w_null[k+1]*f_bump(x,mu_null[k],sigma_null[k])
#            
#        pi_alt = w_alt[0]*f_slope(x,a_alt)
#        for k in range(K):
#            pi_alt += w_alt[k+1]*f_bump(x,mu_alt[k],sigma_alt[k])
#        
#        if len(x.shape)==1:
#            x_idx = x.argsort()
#            plt.figure(figsize=[18,5])
#            plt.plot(x[x_idx],pi_null[x_idx],label='null')
#            plt.plot(x[x_idx],pi_alt[x_idx],label='alt')
#            plt.plot(x[x_idx],(pi_alt[x_idx]+1e-5)/(pi_null[x_idx]+1e-5),label='ratio')
#            
#            bins=np.linspace(0,1,101)
#            p_alt,_= np.histogram(x_alt,bins=bins)
#            p_alt = p_alt/x_alt.shape[0]
#            p_null,_ = np.histogram(x_null,bins=bins)    
#            p_null = p_null/x_null.shape[0]
#            plt.plot(bins[0:-1],p_alt/p_null,label='true_ratio')
#            plt.legend()
#            plt.show()
#            
#        
#        #t = 1/beta*np.log((t_alt+0.00001)/(t_null+0.00001))
#        #t-=t.min()
#        #t = 1/100*np.log((t_alt+0.001)/(t_null+0.001))      
#        t = (pi_alt+1e-5)/(pi_null+1e-5)
#        #t = np.log((pi_alt+pi_null)/(pi_null))
#        
#        
#        
#        gamma = rescale_mirror(t,p,alpha)   
#        t *= gamma
#        result_summary(p<t,h) 
#        
#        d=1 if len(x.shape)==1 else x.shape[1]
#        rand_idx = p<2*t.max()
#        x = x[rand_idx] if d==1 else x[rand_idx,:]
#        p = p[rand_idx]
#        t = t[rand_idx]
#        if h is not None: h = h[rand_idx]
#        rand_idx=np.random.permutation(x.shape[0])[0:min(10000,x.shape[0])]
#        x = x[rand_idx] if d==1 else x[rand_idx,:]
#        p = p[rand_idx]
#        t = t[rand_idx]
#        if h is not None: h = h[rand_idx]
#        
#        if d==1: 
#            sort_idx=x.argsort()
#            plt.figure(figsize=[18,5])
#            if h is None:
#                plt.scatter(x,p,alpha=0.1)
#            else:
#                plt.scatter(x[h==0],p[h==0],alpha=0.1,color='royalblue')
#                plt.scatter(x[h==1],p[h==1],alpha=0.1,color='orange')
#            plt.plot(x[sort_idx],t[sort_idx],color='lime')
#            plt.ylim([t.min()-1e-4,2*t.max()])
#            plt.show()
#        elif d<=3:
#            plt.figure(figsize=[18,5])
#            for i in range(d):
#                plt.subplot('1'+str(d)+str(i+1))
#                sort_idx=x[:,i].argsort()
#                if h is None:
#                    plt.scatter(x[:,i],p,alpha=0.1)
#                else:
#                    plt.scatter(x[:,i][h==0],p[h==0],alpha=0.1,color='royalblue')
#                    plt.scatter(x[:,i][h==1],p[h==1],alpha=0.1,color='orange')
#                plt.scatter(x[:,i][sort_idx],t[sort_idx],s=4,alpha=0.2,color='lime')
#                plt.ylim([t.min()-1e-4,2*t.max()])
#            plt.show()
#
#    return w,a,mu,sigma    

### prim_fdr
#def PrimFDR_SGD(p,x,K=2,alpha=0.1,n_itr=5000,h=None,verbose=False, batchsize = 2000):    
#    p_np = p.copy()
#    x_np = x.copy()
#    if len(x_np.shape) == 1:
#        x_np = x_np.reshape(x_np.shape[0], 1)
#        
#     
#    torch.manual_seed(42)
#    d=1 if len(x.shape)==1 else x.shape[1]
#    
#    _,t_bh   = bh(p,alpha=alpha)
#    lambda0  = 1/t_bh
#    lambda1  = 10/alpha
#    loss_rec = np.zeros([n_itr],dtype=float)
#    n_samp   = x.shape[0]
#    
#    if verbose:
#        print('### Parameters')
#        print('n_itr=%s, n_samp=%s, lambda0=%s, lambda1=%s'%(str(n_itr),str(n_samp),str(lambda0),str(lambda1)))
#    
#    
#    ## calculating the initialization value    
#    w_init,a_init,mu_init,sigma_init = ML_init(p,x,K,alpha=alpha,verbose=True) 
#
#    ## transforming the parameters in the probalistic model to the threshold
#    a     = a_init
#    b     = np.log(w_init[0]*(a_init/(np.exp(a_init)-1)).prod())    
#    w     = np.log(w_init[1:]/((2*np.pi)**(d/2)*sigma_init.prod(axis=1)))
#    mu    = mu_init 
#    sigma = 2*sigma_init**2
#    
#    t = t_cal(x,a,b,w,mu,sigma)
#    
#    gamma=rescale_naive(t,p,alpha*0.9)
#    b += np.log(gamma)
#    w += np.log(gamma)    
#    
#    if verbose:
#        t = rescale_mirror(t,p,alpha)
#        if h is not None:
#            print('## Summary before optimization ##')
#            result_summary(p<t,h)
#        else: 
#            print('## # of discovery: %s'%(np.sum(p<t)))
#    
#    ## initialization  
#    lambda0 = Variable(torch.Tensor([lambda0]),requires_grad=False)
#    lambda1 = Variable(torch.Tensor([lambda1]),requires_grad=False)
#    p       = Variable(torch.zeros(batchsize, ),requires_grad=False)
#    x       = Variable(torch.zeros(batchsize, x_np.shape[1]).float(),requires_grad=False)
#    if d == 1:
#        a = Variable(torch.Tensor([a]),requires_grad=True)
#    else:
#        a = Variable(torch.Tensor(a),requires_grad=True)
#        
#    b       = Variable(torch.Tensor([b]),requires_grad=True)
#    w       = Variable(torch.Tensor(w),requires_grad=True)
#    mu      = Variable(torch.Tensor(mu),requires_grad=True)
#    sigma   = Variable(torch.Tensor(sigma),requires_grad=True)    
#    
#    ## 
#    if verbose:
#        print('### initialization value ###')
#        print ('Slope: a=%s, b=%s'%(str(a.data.numpy()),str(b.data.numpy())))
#        for k in range(K):
#            print('Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w.data.numpy()[k]),mu.data.numpy()[k],sigma.data.numpy()[k]))
#        #print ('w0: %s, mu0: %s, sigma0: %s'%(str(w.data.numpy()[0]),str(mu.data.numpy()[0]),str(sigma.data.numpy()[0])))
#        #print ('w1: %s, mu1: %s, sigma1: %s\n'%(str(w.data.numpy()[1]),str(mu.data.numpy()[1]),str(sigma.data.numpy()[1])))
#        print('\n')
#        
#    optimizer = torch.optim.Adam([a,b,w,mu,sigma],lr=0.0005)
#    optimizer.zero_grad()
#    
#    for l in range(n_itr):
#        
#        p.data.copy_(torch.from_numpy(p_np[np.random.choice(x.shape[0], batchsize)]).float())
#        x.data.copy_(torch.from_numpy(x_np[np.random.choice(x.shape[0], batchsize)]).float())
#        
#        ## calculating the model
#        optimizer.zero_grad()
#        #t = torch.exp(x*a+b)       
#        if d==1:
#            t = torch.exp(x*a+b) 
#        else:
#            t = torch.exp(torch.matmul(x,a)+b)
#        sigma = sigma.clamp(min=1e-6)
#        for i in range(K):
#            if d==1:
#                t = t+torch.exp(w[i]-(x-mu[i])**2/sigma[i])
#            else:
#                t = t+torch.exp(w[i]-torch.matmul((x-mu[i,:])**2,1/sigma[i,:]))
#        loss1 = -torch.mean(torch.sigmoid(lambda0*(t-p)))
#        #loss2 = torch.exp(lambda1*(torch.mean(t)-alpha*torch.mean(torch.sigmoid(lambda0*(t-p)))))
#        loss2 = lambda1*(torch.mean(t)-alpha*torch.mean(torch.sigmoid(lambda0*(t-p)))).clamp(min=0)
#        loss  = loss1+loss2
#        
#        ## backprop
#        loss.backward()
#        optimizer.step()
#        
#        ## show the result 
#        loss_rec[l] = loss.data.numpy()
#        
#        if verbose:
#            if l%(int(n_itr)/5)==0:
#                print('### iteration %s ###'%str(l))
#                print('mean t: ', np.mean(t.data.numpy()))
#                print('mean discovery: ', np.mean(t.data.numpy()>p.data.numpy()))
#                print('loss1: ',loss1.data.numpy())
#                print('loss2: ',loss2.data.numpy())
#                print('n_rej: ',np.sum(t.data.numpy()>p.data.numpy()))
#                print('Estimated FDP: %s'%str((torch.mean(t)/torch.mean(torch.sigmoid(lambda0*(t-p)))).data.numpy()))
#                print('FDP: %s'%str( np.sum((h==0)*(p.data.numpy()<t.data.numpy()))/np.sum(p.data.numpy()<t.data.numpy())))               
#                print ('Slope: a=%s, b=%s'%(str(a.data.numpy()),str(b.data.numpy())))
#                for k in range(K):
#                    print('Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w.data.numpy()[k]),mu.data.numpy()[k],sigma.data.numpy()[k]))
#         
#                if d==1:
#                    plt.figure()
#                    plot_t(t.data.numpy(),x.data.numpy())
#                    plt.show()
#                print('\n')
#    
#    
#    plt.figure()
#    plt.plot(loss_rec)
#    plt.show()
#    p = p.data.numpy()
#    t = rescale_mirror(t.data.numpy(),p,alpha)
#    n_rej=np.sum(p<t)    
#        
#    return n_rej,t
