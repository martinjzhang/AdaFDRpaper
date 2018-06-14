import numpy as np
import scipy as sp
from scipy import stats
from sklearn.mixture import GaussianMixture
import torch
from torch.autograd import Variable
from nfdr2.util import *
from multiprocessing import Pool
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as tf
#from torch.multiprocessing import Pool

np.set_printoptions(precision=4,suppress=True)

""" 
    preprocessing: standardize the hypothesis features 
    
    ----- input  -----
    x_: np_array, n*d, the hypothesis features.
    qt_norm: bool, if perform quantile normalization.
    return_metainfo: bool, if return the meta information regarding the features
    vis_dim: list, the dimensions to visualize. counting starts from 0, needed only when verbose is True
    verbose: bool, if generate ancillary information
    
    ----- output -----
    x: Processed feature stored as an n*d array. The discrete feature is reordered based on the alt/null ratio.
    meta_info: a d*2 array. The first dimensional corresponds to the type of the feature (continuous/discrete). The second 
               dimension is a list on the mapping (from small to large). For example, if the before reorder is [0.1,0.2,0.7].
               This list may be [0.7,0.1,0.2].
""" 

def feature_preprocess(x_,p,qt_norm=True,continue_rank=True,require_meta_info=False,vis_dim=None,verbose=False):   
    x = x_.copy()
    if len(x.shape) == 1: x = x.reshape([-1,1])
    _,d = x.shape
        
    ## feature visualization before preprocessing
    if verbose:
        plt.figure(figsize=[18,5])
        plt.suptitle('before feature_preprocess')
        plot_x(x,vis_dim=vis_dim)       
        plt.show()                
    
    ## preprocesing
    # reorder the discrete features as well as calculating the meta information
    meta_info = []
    for i in range(d):
        x[:,i],temp_meta = cal_meta_info(x[:,i],p)
        meta_info.append(temp_meta)
        
    # quantile normalization
    if qt_norm: 
        x=rank(x,continue_rank=continue_rank)
               
    # scale to be between 0 and 1
    x = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)) 
    
    if verbose:
        plt.figure(figsize=[18,5])
        plt.suptitle('after feature_preprocess')
        plot_x(x,vis_dim=vis_dim) 
        plt.show()
    
    if require_meta_info:
        return x,meta_info
    else:
        return x
    
def cal_meta_info(x,p):
    if np.unique(x).shape[0]<100:
        x_new,x_order_new = reorder_discrete_feature(x,p) 
        return x_new,['discrete',x_order_new]
    else: 
        return x,['continuous',None]  
        
def reorder_discrete_feature(x,p):
    ## separate the null and the alt proportion
    _,t_BH = bh(p,alpha=0.1) # fixit: where is n_full
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

""" 
    feature_explore: provide a visualization of pi1/pi0 for each dimension, to visualize the amount of information carried by each dimension
    ----- input  -----
    
    ----- output -----
    
"""
def feature_explore(p,x_,alpha=0.1,n_full=None,vis_dim=None,cate_name={},output_folder=None,h=None,log_transform=False):
    def plot_feature_1d(x_margin,p,x_null,x_alt,meta_info,title='',cate_name=None,\
                        output_folder=None,h=None):
        feature_type,cate_order = meta_info       
        x_min,x_max = np.percentile(x_margin,[1,99])
        x_range = x_max-x_min
        x_min -= 0.05*x_range
        x_max += 0.05*x_range
        
        bins = np.linspace(x_min,x_max,101)
        bin_width = bins[1]-bins[0]
        
        if feature_type == 'continuous':         
            ## continuous feature: using kde to estimate 
            n_bin = bins.shape[0]-1
            x_grid = (bins+bin_width/2)[0:-1]
            p_null,_ = np.histogram(x_null,bins=bins) 
            p_alt,_= np.histogram(x_alt,bins=bins)         
            p_null = p_null/np.sum(p_null)*n_bin
            p_alt = p_alt/np.sum(p_alt)*n_bin
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
            for i,key in enumerate(unique_null): 
                p_null[unique_val==key] = cts_null[i]                
            for i,key in enumerate(unique_alt): 
                p_alt[unique_val==key] = cts_alt[i]           
            n_bin = unique_val.shape[0]           
            p_null = (p_null+1)/np.sum(p_null+1)*n_bin
            p_alt = (p_alt+1)/np.sum(p_alt+1)*n_bin            
            p_ratio = (p_alt+1e-2)/(p_null+1e-2)  
            x_grid = (np.arange(unique_val.shape[0])+1)/(unique_val.shape[0]+1)
            x_min,x_max,bin_width = 0,1,1/(unique_val.shape[0]+1)
            
            if cate_name is None: 
                cate_name_ = cate_order
            else:
                cate_name_ = []
                for i in cate_order:
                    cate_name_.append(cate_name[i])
                    
        plt.figure(figsize=[8,8])
        plt.subplot(311)
        rnd_idx=np.random.permutation(p.shape[0])[0:np.min([10000,p.shape[0]])]
        p = p[rnd_idx]
        x_margin = x_margin[rnd_idx]
       
        if h is not None:
            h = h[rnd_idx]
            plt.scatter(x_margin[h==1],p[h==1],color='orange',alpha=0.3,s=4,label='alt')
            plt.scatter(x_margin[h==0],p[h==0],color='royalblue',alpha=0.3,s=4,label='null')
        else:
            plt.scatter(x_margin,p,color='royalblue',alpha=0.3,s=4,label='alt')
        plt.ylim([0,(p[p<0.5].max())])
        plt.ylabel('p-value')
        plt.title(title+' (%s)'%feature_type)
        plt.subplot(312)
        plt.bar(x_grid,p_null,width=bin_width,color='royalblue',alpha=0.6,label='null')
        plt.bar(x_grid,p_alt,width=bin_width,color='darkorange',alpha=0.6,label='alt')
        plt.xlim([x_min,x_max])
        if feature_type=='discrete': 
            plt.xticks(x_grid,cate_name_,rotation=45)
        plt.ylabel('null/alt proportion')
        plt.legend()
        plt.subplot(313)
        if feature_type == 'continuous':
            plt.plot(x_grid,p_ratio,color='seagreen',label='ratio',linewidth=4) 
        else:
            plt.plot(x_grid,p_ratio,color='seagreen',marker='o',label='ratio',linewidth=4)
            plt.xticks(x_grid,cate_name_,rotation=45)
        plt.xlim([x_min,x_max])
        plt.ylabel('ratio')
        plt.xlabel('Covariate $x$')
        plt.tight_layout()
        if output_folder is not None:
            plt.savefig(output_folder+'/explore_%s.png'%title)
        plt.show()
        
    x = x_.copy()
    if n_full is None: 
        n_full = p.shape[0]
    if len(x.shape) == 1: 
        x = x.reshape([-1,1])
    _,d = x.shape
    
    ## some transformation
    meta_info = []
    for i in range(d):
        x[:,i],temp_meta = cal_meta_info(x[:,i],p)
        meta_info.append(temp_meta)
        
    if log_transform:
        for i in range(d):
            if meta_info[i][0]=='continuous':
                x[:,i] = -np.log10(x[:,i].clip(min=1e-20))
    
    ## separate the null proportion and the alternative proportion
    _,t_BH = bh(p,n_full=n_full,alpha=0.1)
    x_null,x_alt = x[p>0.75],x[p<t_BH]      
    
    ## generate the figure
    if vis_dim is None: 
        vis_dim = np.arange(min(4,d))    
    for i in vis_dim:
        x_margin = x[:,i]
        temp_null,temp_alt = x_null[:,i],x_alt[:,i]
        if i in cate_name.keys():
            plot_feature_1d(x_margin,p,temp_null,temp_alt,meta_info[i],title='feature_%s'%str(i+1),\
                            cate_name=cate_name[i],output_folder=output_folder,h=h)
        else:
            plot_feature_1d(x_margin,p,temp_null,temp_alt,meta_info[i],title='feature_%s'%str(i+1),\
                            output_folder=output_folder,h=h)
    return

#def feature_explore(p,x_,alpha=0.1,qt_norm=False,vis_dim=None,cate_name={},output_folder=None,h=None):
#    def plot_feature_1d(x_margin,p,x_null,x_alt,bins,meta_info,title='',cate_name=None,\
#                        output_folder=None,h=None):
#        feature_type,cate_order = meta_info        
#        if feature_type == 'continuous':         
#            ## continuous feature: using kde to estimate 
#            n_bin = bins.shape[0]-1
#            x_grid = (bins+(bins[1]-bins[0])/2)[0:-1]
#            p_null,_ = np.histogram(x_null,bins=bins) 
#            p_alt,_= np.histogram(x_alt,bins=bins)         
#            p_null = p_null/np.sum(p_null)*n_bin
#            p_alt = p_alt/np.sum(p_alt)*n_bin
#            kde_null = stats.gaussian_kde(x_null).evaluate(x_grid)
#            kde_alt = stats.gaussian_kde(x_alt).evaluate(x_grid)
#            p_ratio = (kde_alt+1e-2)/(kde_null+1e-2)        
#                 
#        else: 
#            ## discrete feature: directly use the empirical counts 
#            unique_null,cts_null = np.unique(x_null,return_counts=True)
#            unique_alt,cts_alt = np.unique(x_alt,return_counts=True)            
#            unique_val = np.array(list(set(list(unique_null)+list(unique_alt))))
#            unique_val = np.sort(unique_val)            
#            p_null,p_alt = np.zeros([unique_val.shape[0]]),np.zeros([unique_val.shape[0]])          
#            for i,key in enumerate(unique_null): 
#                p_null[unique_val==key] = cts_null[i]                
#            for i,key in enumerate(unique_alt): 
#                p_alt[unique_val==key] = cts_alt[i]           
#            n_bin = unique_val.shape[0]           
#            p_null = (p_null+1)/np.sum(p_null+1)*n_bin
#            p_alt = (p_alt+1)/np.sum(p_alt+1)*n_bin            
#            p_ratio = (p_alt+1e-2)/(p_null+1e-2)  
#            x_grid = (np.arange(unique_val.shape[0])+1)/(unique_val.shape[0]+1)
#            
#            if cate_name is None: 
#                cate_name_ = cate_order
#            else:
#                cate_name_ = []
#                for i in cate_order:
#                    cate_name_.append(cate_name[i])
#                    
#        plt.figure(figsize=[8,8])
#        plt.subplot(311)
#        rnd_idx=np.random.permutation(p.shape[0])[0:np.min([10000,p.shape[0]])]
#        p = p[rnd_idx]
#        x_margin = x_margin[rnd_idx]
#        
#        if h is not None:
#            plt.scatter(x_margin[h==1],p[h==1],color='orange',alpha=0.3,s=4,label='alt')
#            plt.scatter(x_margin[h==0],p[h==0],color='royalblue',alpha=0.3,s=4,label='null')
#        else:
#            plt.scatter(x_margin,p,color='royalblue',alpha=0.3,s=4,label='alt')
#        plt.ylim([0,(p[p<0.5].max())])
#        plt.ylabel('p-value')
#        plt.title(title+' (%s)'%feature_type)
#        plt.subplot(312)
#        plt.bar(x_grid,p_null,width=1/n_bin,color='royalblue',alpha=0.6,label='null')
#        plt.bar(x_grid,p_alt,width=1/n_bin,color='darkorange',alpha=0.6,label='alt')
#        plt.xlim([0,1])
#        if feature_type=='discrete': 
#            plt.xticks(x_grid,cate_name_,rotation=45)
#        plt.ylabel('null/alt proportion')
#        plt.legend()
#        plt.subplot(313)
#        if feature_type == 'continuous':
#            plt.plot(x_grid,p_ratio,color='seagreen',label='ratio',linewidth=4) 
#        else:
#            plt.plot(x_grid,p_ratio,color='seagreen',marker='o',label='ratio',linewidth=4)
#            plt.xticks(x_grid,cate_name_,rotation=45)
#        plt.xlim([0,1])
#        plt.ylabel('ratio')
#        plt.xlabel('Covariate $x$')
#        plt.tight_layout()
#        if output_folder is not None:
#            plt.savefig(output_folder+'/explore_%s.png'%title)
#        plt.show()
#    
#    
#    ## preprocessing
#    x,meta_info = feature_preprocess(x_,p,qt_norm=qt_norm,continue_rank=False,require_meta_info=True)   
#    x_nq,meta_info = feature_preprocess(x_,p,qt_norm=False,continue_rank=False,require_meta_info=True)   
#    d = x.shape[1]
#    
#    ## separate the null proportion and the alternative proportion
#    _,t_BH = bh(p,alpha=alpha)
#    x_null,x_alt = x[p>0.5],x[p<t_BH]      
#    x_null_nq,x_alt_nq = x_nq[p>0.5],x_nq[p<t_BH]      
#    
#    ## generate the figure
#    bins = np.linspace(0,1,26)  
#    if vis_dim is None: 
#        vis_dim = np.arange(min(4,d))    
#    
#    for i in vis_dim:
#        x_margin = x[:,i]
#        if meta_info[i][0] == 'continuous':
#            temp_null,temp_alt = x_null[:,i],x_alt[:,i]
#        else:
#            temp_null,temp_alt = x_null_nq[:,i],x_alt_nq[:,i]
#        if i in cate_name.keys():
#            plot_feature_1d(x_margin,p,temp_null,temp_alt,bins,meta_info[i],title='feature_%s'%str(i+1),\
#                            cate_name=cate_name[i],output_folder=output_folder,h=h)
#        else:
#            plot_feature_1d(x_margin,p,temp_null,temp_alt,bins,meta_info[i],title='feature_%s'%str(i+1),\
#                            output_folder=output_folder,h=h)
#    return
              
       
"""
    PrimFDR: the main testing function 
"""

def pfdr_test(data):  
    print('pfdr_test start')
    p1,x1 = data[0]
    p2,x2 = data[1]
    K,alpha,n_full,n_itr,output_folder,logger,random_state = data[2]
    fold_number = data[3]
    
    fname = output_folder+'/record_fold_%d.txt'%fold_number
    f_write = open(fname,'w+')
    f_write.write('### record for fold_%d\n'%fold_number)
        
    print('PrimFDR start')
    _,_,theta = PrimFDR(p1,x1,K=K,alpha=alpha,n_full=n_full,n_itr=n_itr,verbose=True,\
                        output_folder=output_folder,logger=logger,fold_number=fold_number,\
                        random_state=random_state,f_write=f_write)
    a,b,w,mu,sigma,gamma = theta
    
    t2 = t_cal(x2,a,b,w,mu,sigma)
    gamma = rescale_mirror(t2,p2,alpha,f_write=f_write,title='cv')
    t2 = gamma*t2
    
    if f_write is not None:
        f_write.write('\n## Test result with PrimFDR_cv fold_%d\n'%fold_number)
        result_summary(p2<t2,None,f_write=f_write)
    f_write.close()
    return np.sum(p2<t2),t2,[a,b,w,mu,sigma,gamma]

def PrimFDR_cv(p,x,K=3,alpha=0.1,n_full=None,n_itr=1000,qt_norm=True,h=None,\
               verbose=False,output_folder=None,logger=None,random_state=0):
    np.random.seed(random_state)
    if len(x.shape) == 1: 
        x = x.reshape([-1,1])
    n_sample,d = x.shape
    #x = feature_preprocess(x,p,qt_norm=qt_norm)
    
    ## random split the fold
    n_sub = int(n_sample/2)
    rand_idx = np.random.permutation(n_sample)
    n_full = int(n_full/2)
    
    fold_idx = np.zeros([n_sample],dtype=int)
    fold_idx[rand_idx[0:n_sub]] = 0
    fold_idx[rand_idx[n_sub:]] = 1
    
    # 10 layers
        
    if verbose:
        start_time=time.time()
        if logger is None:
            print('#time start: 0.0s')
        else:
            logger.info('#time start: 0.0s')

    ## construct the data
    args = [K,alpha,n_full,n_itr,output_folder,None,random_state]
    data = {}
    for i in range(2): 
        data[i] = [p[fold_idx==i],x[fold_idx==i]]
            
    Y_input = []    
    Y_input.append([data[1],data[0],args,0])
    Y_input.append([data[0],data[1],args,1])
    if verbose: 
        print('#time input: %0.4fs'%(time.time()-start_time))
        logger.info('#time input: %0.4fs'%(time.time()-start_time))

    po = Pool(2)
    res = po.map(pfdr_test, Y_input)

    if verbose: 
        print('#time test: %0.4fs'%(time.time()-start_time))
        logger.info('#time test: %0.4fs'%(time.time()-start_time))
    
    ## Summarize the result
    theta = []
    n_rej = []    
    t = np.zeros([n_sample],dtype=float)
     
    for i in range(2):
        n_rej.append(res[i][0])
        t[fold_idx==i] = res[i][1]
        theta.append(res[i][2])
    
    if verbose:
        color_list = ['navy','orange']

        print('# total rejection: %d'%np.array(n_rej).sum(), n_rej)
        #logger.info('# total rejection: %d'%np.array(n_rej).sum(), n_rej)
        if h is not None:
            tol_rej = np.sum(p<t)
            false_rej = np.sum((p<t)*(h==0))
            
            logger.info('## Testing summary (ground truth) ##')
            logger.info('# D=%d, FD=%d, FDP=%0.3f'%(tol_rej,false_rej,false_rej/tol_rej))
        
        plt.figure(figsize=[8,12])
        n_figure = min(d,4)

        for i_dim in range(n_figure):
            plt.subplot(str(n_figure)+'1'+str(i_dim+1))
            for i in range(2):
                if h is not None:
                    plot_scatter_t(t[fold_idx==i],p[fold_idx==i],x[fold_idx==i,i_dim],\
                                   h[fold_idx==i],color=color_list[i],label='fold '+str(i+1))
                else:
                    plot_scatter_t(t[fold_idx==i],p[fold_idx==i],x[fold_idx==i,i_dim],\
                                   h,color=color_list[i],label='fold '+str(i+1))
            plt.legend()
            plt.title('Dimension %d'%(i_dim+1))

        if output_folder is not None:
            plt.savefig(output_folder+'/learned_threshold.png')
        else:
            plt.show()
        print('#time total: %0.4fs'%(time.time()-start_time))     
    
    return n_rej,t,theta   

def PrimFDR(p,x,K=5,alpha=0.1,n_full=None,n_itr=1500,qt_norm=True,h=None,verbose=False,debug='',\
            output_folder=None,logger=None,fold_number=0,random_state=0,f_write=None):   
     
    ## feature preprocessing 
    torch.manual_seed(random_state)
    if len(x.shape)==1:
        x = x.reshape([-1,1])
    d = x.shape[1]
    #x = feature_preprocess(x,p,qt_norm=qt_norm)
    
    if f_write is not None:
        f_write.write('# n_sample=%d\n'%(x.shape[0]))
        for i in [1e-6,1e-5,1e-4,5e-4]:
            f_write.write('# p<%0.6f: %d\n'%(i,np.sum(p<i)))
        f_write.write('\n')
    
    # rough threshold calculation using PrimFDR_init 
    w_init,a_init,mu_init,sigma_init = PrimFDR_init(p,x,K,alpha=alpha,n_full=n_full,verbose=verbose,\
                                                    output_folder=output_folder,logger=logger,\
                                                    random_state=random_state,fold_number=fold_number,\
                                                    f_write=f_write) 

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
    gamma=rescale_mirror(t,p,alpha,f_write=f_write,title='before optimization')
    t = gamma*t
    b,w = b+np.log(gamma),w+np.log(gamma) 
    
    ## optimization: parameter setting 
    # lambda0: adaptively set based on the approximation accuracy of the sigmoid function
    lambda0,n_rej,n_fd = 1/t.mean(),np.sum(p<t),np.sum(p>1-t)
    
    if f_write is not None:
        f_write.write('\n## choosing lambda0\n')
        f_write.write('## lambda0=%0.1f, D=%d, FD_hat=%d, alpha_hat=%0.3f\n'%(lambda0,n_rej,n_fd,n_fd/n_rej))
        
    while np.absolute(np.sum(sigmoid((lambda0*(t-p))))-n_rej)>0.02*n_rej \
        or np.absolute(np.sum(sigmoid((lambda0*(t+p-1))))-n_fd)>0.02*n_fd:
        lambda0 = lambda0+0.5/t.mean()
        if f_write is not None:
            f_write.write('# lambda0=%0.1f, D_apr=%0.1f (r err=%0.3f), FD_hat_apr=%0.3f (r err=%0.3f)\n'\
                          %(lambda0,np.sum(sigmoid((lambda0*(t-p)))),np.absolute(1-np.sum(sigmoid((lambda0*(t-p))))/n_rej),\
                            np.sum(sigmoid((lambda0*(t+p-1)))),np.absolute(1-np.sum(sigmoid((lambda0*(t+p-1))))/n_fd)))
    
    # choose other parameters 
    lambda1  = 10/alpha # 100
    lambda0,lambda1 = int(lambda0),int(lambda1)
    loss_rec = np.zeros([n_itr],dtype=float)
    n_samp   = x.shape[0]
    if verbose: 
        print('## optimization paramter:')
        print('# n_itr=%d, n_samp=%d, lambda0=%d, lambda1=%d'%(n_itr,n_samp,lambda0,lambda1))
        if logger is not None:
            logger.info('## fold_%d: optimization paramter:'%fold_number)
            logger.info('# n_itr=%s, n_samp=%s, lambda0=%s, lambda1=%s\n'%(str(n_itr),str(n_samp),str(lambda0),str(lambda1)))
        if f_write is not None:
            f_write.write('\n## optimization paramter:\n')
            f_write.write('# n_itr=%d, n_samp=%d, lambda0=%0.4f, lambda1=%0.4f\n'%(n_itr,n_samp,lambda0,lambda1))
           
    ## optimization: initialization         
    lambda0 = Variable(torch.Tensor([lambda0]),requires_grad=False)
    lambda1 = Variable(torch.Tensor([lambda1]),requires_grad=False)
    p       = Variable(torch.from_numpy(p).float(),requires_grad=False)
    #x       = Variable(torch.from_numpy(x).float().view(-1,d),requires_grad=False)
    x       = Variable(torch.from_numpy(x).float(),requires_grad=False)
    a       = Variable(torch.Tensor(a),requires_grad=True)
    b       = Variable(torch.Tensor([b]),requires_grad=True)
    w       = Variable(torch.Tensor(w),requires_grad=True)
    mu      = Variable(torch.Tensor(mu),requires_grad=True)
    #sigma = sigma.clip(min=1e-6)
    #sigma_mean = np.mean(1/sigma)
    #print(sigma_mean)
    #sigma   = Variable(torch.Tensor((1/sigma) / (sigma_mean)),requires_grad=True)    
    
    sigma_mean = np.mean(sigma,axis=0)
    #sigma_mean = np.mean(sigma)
    #print(sigma_mean)
    sigma   = Variable(torch.Tensor(sigma/sigma_mean),requires_grad=True)
    sigma_mean = Variable(torch.from_numpy(sigma_mean).float(),requires_grad=False)
    
    if verbose:
        #print('## optimization initialization:')
        #print ('# Slope: a=%s, b=%s'%(str(a.data.numpy()),str(b.data.numpy())))
        #for k in range(K):
        #    print('# Bump %s: w=%s, mu=%s, sigma=%s'%(str(k),str(w.data.numpy()[k]),mu.data.numpy()[k],sigma.data.numpy()[k]))
        #print('\n')
        
        if logger is not None:
            logger.info('## optimization initialization:')
            logger.info ('# Slope: a=%s, b=%s'%(str(a.data.numpy()),str(b.data.numpy())))
            for k in range(K):
                logger.info('# Bump %s: w=%s, mu=%s, sigma=%s'\
                            %(str(k),str(w.data.numpy()[k]),mu.data.numpy()[k],sigma.data.numpy()[k]))
            logger.info('\n')
            
        if f_write is not None:
            f_write.write('\n## PrimFDR: initialization\n')
            f_write.write('# Slope: a=%s\n'%(a.data.numpy()))
            f_write.write('         b=%0.4f\n'%(b.data.numpy()))
            for k in range(K):
                f_write.write('# Bump %d: w=%0.4f\n'%(k,w.data.numpy()[k]))
                f_write.write('         mu=%s\n'%(mu.data.numpy()[k])) 
                f_write.write('      sigma=%s\n'%(sigma.data.numpy()[k])) 
            f_write.write('\n')
            
    ## 1    
    #optimizer = torch.optim.Adam([a,b,w,mu,sigma],lr=0.02)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,300,600],gamma=0.5)
    
    ## 2
    #optimizer = torch.optim.Adam([a,b,w,mu,sigma],lr=0.05)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,300,600],gamma=0.25)
    
    ## 3
    #optimizer = torch.optim.Adam([a,b,w,mu,sigma],lr=0.02)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[300],gamma=0.1)
    
    ## 4
    #optimizer = torch.optim.Adam([a,b,w,mu,sigma],lr=0.05)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200],gamma=0.05)
    
    optimizer = torch.optim.Adam([a,b,w,mu,sigma],lr=0.02)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,300,600],gamma=0.5)     
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
                #print('## iteration %s'%str(l))    
                #print('n_rej: ',np.sum(t.data.numpy()>p.data.numpy()))
                #print('n_rej sig: ',np.sum(sigmoid(lambda0.data.numpy()*(t.data.numpy()-p.data.numpy()))))
                #print('FD esti mirror:',np.sum(p.data.numpy()>1-t.data.numpy()))
                #print('FD esti mirror sig:',np.sum(sigmoid(lambda0.data.numpy()*(p.data.numpy()-1+t.data.numpy()))))             
                #print('loss1: ',loss1.data.numpy())
                #print('loss2: ',loss2.data.numpy())
                #print('Estimated FDP: %s'\
                #      %str((torch.mean(torch.sigmoid(lambda0*(t+p-1)))/torch.mean(torch.sigmoid(lambda0*(t-p)))).data.numpy()))
                #print('FDP: %s'%str( np.sum((h==0)*(p.data.numpy()<t.data.numpy()))/np.sum(p.data.numpy()<t.data.numpy())))
                #print('Slope: a=%s, b=%s'%(str(a.data.numpy()),str(b.data.numpy())))
                #for k in range(K):
                #    print('Bump %s: w=%s, mu=%s, sigma=%s'\
                #          %(str(k),str(w.data.numpy()[k]),mu.data.numpy()[k],sigma.data.numpy()[k]))
                
                if f_write is not None:
                    f_write.write('\n## iteration %d: \n'%(l))    
                    f_write.write('# n_rej=%d, n_rej_apr=%0.1f, FD_hat=%d, FD_hat_apr=%0.1f\n'%\
                                  (np.sum(t.data.numpy()>p.data.numpy()),\
                                   np.sum(sigmoid(lambda0.data.numpy()*(t.data.numpy()-p.data.numpy()))),\
                                   np.sum(p.data.numpy()>1-t.data.numpy()),\
                                   np.sum(sigmoid(lambda0.data.numpy()*(p.data.numpy()-1+t.data.numpy())))))
                    f_write.write('# loss1=%0.4f, loss2=%0.4f, FDP=%0.4f, FDP_hat_apr=%0.4f\n'\
                                  %(loss1.data.numpy(),loss2.data.numpy(),\
                                  np.sum((h==0)*(p.data.numpy()<t.data.numpy()))/np.sum(p.data.numpy()<t.data.numpy()),\
                                  (torch.mean(torch.sigmoid(lambda0*(t+p-1)))\
                                  /torch.mean(torch.sigmoid(lambda0*(t-p)))).data.numpy()))
                    f_write.write('# Slope: a=%s\n'%(a.data.numpy()))
                    f_write.write('         b=%0.4f\n'%(b.data.numpy()))
                    for k in range(K):
                        f_write.write('# Bump %d: w=%0.4f\n'%(k,w.data.numpy()[k]))
                        f_write.write('         mu=%s\n'%(mu.data.numpy()[k])) 
                        f_write.write('      sigma=%s\n'%(sigma.data.numpy()[k])) 
                    
                                                                      
                if d==1:
                    plt.figure(figsize=[8,5])
                    plot_t(t.data.numpy(),p.data.numpy(),x.data.numpy(),h)
                    if output_folder is not None:
                        plt.savefig(output_folder+'/threshold_itr_%d_fold_%d.png'%(l,fold_number))
                    else:
                        plt.show()
                #print('\n')
                
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
        plt.figure(figsize=[6,5])
        plt.plot(np.log(loss_rec-loss_rec.min()+1e-3))
        if output_folder is not None:
            plt.savefig(output_folder+'/loss_fold_%d.png'%fold_number)
        else:
            plt.show()       
        
    p = p.data.numpy()
    x = x.data.numpy()
    
    a,b,w,mu,sigma = a.data.numpy(),b.data.numpy(),w.data.numpy(),mu.data.numpy(),(sigma * sigma_mean).data.numpy()

    t = t_cal(x,a,b,w,mu,sigma)
    gamma = rescale_mirror(t,p,alpha,f_write=f_write,title='after PrimFDR')   
    t *= gamma
    n_rej=np.sum(p<t)     
    if verbose: 
        if f_write is not None:
            f_write.write('\n## Test result with PrimFDR\n')
        result_summary(p<t,h,f_write=f_write)
    theta = [a,b,w,mu,sigma,gamma]  
    return n_rej,t,theta

"""
    initialization function of PrimFDR: fit the mixture model with a linear trend and a Gaussian mixture 
"""
def PrimFDR_init(p,x,K,alpha=0.1,n_full=None,n_itr=100,h=None,verbose=False,output_folder=None,\
                 logger=None,random_state=0,fold_number=0,f_write=None):
    #x = feature_preprocess(x,p)
    np.random.seed(random_state)
    if verbose: 
        print('## PrimFDR_init starts')   
    if logger is not None:
        logger.info('## PrimFDR_init starts')  
    if f_write is not None:
        f_write.write('## PrimFDR_init starts\n')
        
    if len(x.shape)==1: 
        x = x.reshape([-1,1])
    n_samp,d = x.shape
    if n_full is None:
        n_full = n_samp
            
    ## extract the null and the alternative proportion
    _,t_BH = bh(p,n_full=n_full,alpha=0.1)
    x_null,x_alt = x[p>0.75],x[p<t_BH]
    
    if f_write is not None:
        f_write.write('# t_BH=%0.6f, n_null=%d, n_alt=%d\n'%(t_BH, x_null.shape[0],x_alt.shape[0]))
    
    
    ## fit the null distribution
    if verbose: 
        print('## Learning null distribution')
    if logger is not None: 
        logger.info('## Learning null distribution')   
    if f_write is not None:
        f_write.write('## Learning null distribution\n')    
        
    w_null,a_null,mu_null,sigma_null = mixture_fit(x_null,K,n_itr=n_itr,verbose=verbose,\
                                                   logger=logger,output_folder=output_folder,\
                                                   suffix='_null',random_state=random_state,\
                                                   fold_number=fold_number,f_write=f_write)   
    
    x_w = 1/(f_all(x_alt,a_null,mu_null,sigma_null,w_null)+1e-5)
    x_w /= np.mean(x_w)
    
    if verbose: 
        print('## Learning alternative distribution')
    if logger is not None: 
        logger.info('## Learning alternative distribution')
    if f_write is not None:
        f_write.write('## Learning alternative distribution\n')  
    w,a,mu,sigma = mixture_fit(x_alt,K,x_w=x_w,n_itr=n_itr,verbose=verbose,\
                               logger=logger,output_folder=output_folder,\
                               suffix='_alt',random_state=random_state,\
                               fold_number=fold_number,f_write=f_write)
    
    if verbose:        
        t = f_all(x,a,mu,sigma,w)
        gamma = rescale_mirror(t,p,alpha)   
        t = t*gamma
        print('## Test result with PrimFDR_init')
        if logger is not None: 
            logger.info('## Test result with PrimFDR_init')
        if f_write is not None:
            f_write.write('\n## Test result with PrimFDR_init\n')
        
        result_summary(p<t,h,logger=logger,f_write=f_write) 
        
        if d==1:
            plt.figure(figsize=[8,5])
        else:
            plt.figure(figsize=[8,12])
        plot_t(t,p,x,h)
        if output_folder is not None: 
            plt.savefig(output_folder+'/threshold_after_PrimFDR_init_fold_%d.png'%fold_number)
        else:
            plt.show()
        print('## PrimFDR_init finishes')
        if logger is not None: 
            logger.info('## PrimFDR_init finishes')
        if f_write is not None:
            f_write.write('## PrimFDR_init finished\n')
    return w,a,mu,sigma    

""" 
    mixture_fit: fit a GLM+Gaussian mixture using EM algorithm
"""
def mixture_fit(x,K=3,x_w=None,n_itr=1000,verbose=False,debug=False,logger=None,output_folder=None,\
                suffix=None,random_state=0,fold_number=0,f_write=None):   
    np.random.seed(random_state)
    if len(x.shape)==1: 
        x = x.reshape([-1,1])
    n_samp,d = x.shape
    if x_w is None: 
        x_w=np.ones([n_samp],dtype=float)
        
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
        #print('# Slope: w=%0.4f, a=%0.4f'%(w[0],a))
        #for k in range(K):
        #    print('# Bump %d: w=%0.4f'%(k,w[k+1]))
        #    print('         mu=',mu[k]) 
        #    print('      sigma=',sigma[k]) 
        #    print()
        #print('\n')
        
        if f_write is not None:
            f_write.write('## mixture_fit: initialization\n')
            f_write.write('# Slope: w=%0.4f, a=%s\n'%(w[0],a))
            for k in range(K):
                f_write.write('# Bump %d: w=%0.4f\n'%(k,w[k+1]))
                f_write.write('         mu=%s\n'%(mu[k])) 
                f_write.write('      sigma=%s\n'%(sigma[k])) 
            f_write.write('\n')
            
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

    if i >= n_itr and verbose: 
        print('!!! the model does not converge')
        
        if f_write is not None:
            f_write.write('!!! the model does not converge \n')
    
    if verbose: 
        print('## mixture_fit: learned parameters')
        #print('# Slope: w=%0.4f, a=%0.4f'%(w[0],a))
        #for k in range(K):
        #    print('# Bump %d: w=%0.4f'%(k,w[k+1]))
        #    print('         mu=',mu[k]) 
        #    print('      sigma=',sigma[k]) 
            
        if f_write is not None:
            f_write.write('## mixture_fit learned parameters\n')
            f_write.write('# Slope: w=%0.4f, a=%s\n'%(w[0],a))
            for k in range(K):
                f_write.write('# Bump %d: w=%0.4f\n'%(k,w[k+1]))
                f_write.write('         mu=%s\n'%(mu[k])) 
                f_write.write('      sigma=%s\n'%(sigma[k])) 
            f_write.write('\n')
            
    #if d==1 and verbose:        
    #    plt.figure(figsize=[8,5])
    #    plt.subplot(121)
    #    temp_hist,_,_=plt.hist(x,bins=50,weights=1/n_samp*50*np.ones([n_samp]))
    #    temp = np.linspace(0,1,101)
    #    x_grid = temp.reshape([-1,1])
    #    plt.plot(temp,f_all(x_grid,a,mu,sigma,w))
    #    plt.ylim([0,1.5*temp_hist.max()])
    #    plt.title('unweighted')
    #    plt.subplot(122)
    #    temp_hist,_,_=plt.hist(x,bins=50,weights=x_w/n_samp*50)
    #    temp = np.linspace(0,1,101)
    #    plt.plot(temp,f_all(x_grid,a,mu,sigma,w))
    #    plt.ylim([-1e-3,1.5*temp_hist.max()])
    #    plt.title('weighted')
    #    plt.show()   
        
    if output_folder is not None:
        bins_ = np.linspace(0,1,101)
        x_grid = bins_.reshape([-1,1])
        
        if d==1:
            plt.figure(figsize=[8,5])
            plt.hist(x,bins=bins_,weights=x_w/np.sum(x_w)*100) 
            temp_p = f_all(x_grid,a,mu,sigma,w)      
            plt.plot(bins_,temp_p)
            plt.savefig(output_folder+'/projection%s_fold_%d.png'%(suffix,fold_number))
        
        else:
            plt.figure(figsize=[8,12])
            n_figure = min(d,4)
            for i_dim in range(n_figure):        
                plt.subplot(str(n_figure)+'1'+str(i_dim+1))
                plt.hist(x[:,i_dim],bins=bins_,weights=x_w/np.sum(x_w)*100)  
                temp_p = f_all(x_grid,a[[i_dim]],mu[:,[i_dim]],sigma[:,[i_dim]],w)  
                #temp_p = np.zeros(bins_.shape[0])
                #temp_p += w[0]*f_slope(x_grid,a[[i_dim]])
                #for i in range(1,w.shape[0]):
                #    temp_p += w[i]*f_bump(x_grid,mu[i-1,[i_dim]],sigma[i-1,[i_dim]])
                plt.plot(bins_,temp_p)
                plt.title('Dimension %d'%(i_dim+1))
            plt.savefig(output_folder+'/projection%s_fold_%d.png'%(suffix,fold_number))
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
def bh(p,alpha=0.1,n_full=None,verbose=False):
    if n_full is None: 
        n_full = p.shape[0]
    p_sort   = sorted(p)
    n_rej    = 0
    for i in range(p.shape[0]):
        if p_sort[i] < i*alpha/n_full:
            n_rej = i
    t_rej = p_sort[n_rej]
    if verbose:
        print("## bh testing summary ##")
        print("# n_rej = %d"%n_rej)
        print("# t_rej = %0.6f"%t_rej)
        print("\n")
    return n_rej,t_rej

def storey_bh(p,alpha=0.1,lamb=0.5,n_full=None,verbose=False):
    if n_full is None: 
        n_full = p.shape[0]
    else:
        lamb = np.min(p[p>0.5])
        
    pi0_hat  = (np.sum(p>lamb)/(1-lamb)/n_full).clip(max=1)  
    alpha   /= pi0_hat
    print('## pi0_hat=%0.3f'%pi0_hat)
    
    p_sort = sorted(p)
    n_rej = 0
    for i in range(p.shape[0]):
        if p_sort[i] < i*alpha/n_full:
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
