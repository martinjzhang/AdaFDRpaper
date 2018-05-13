import matplotlib
matplotlib.use('Agg')
import logging
import numpy as np
from scipy.stats import norm
import os
import data_loader as dl
import prim_fdr as pf
import time
import matplotlib.pyplot as plt

output_folder = './results/unit_test_mixture_fit'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
logging.basicConfig(level=logging.INFO,
                    format='%(module)s:: %(message)s',
                    filename=output_folder+'/result.txt', filemode='w')
logger = logging.getLogger()
    
def benchmark_mixture_fit(x,output_folder,suffix,logger=None):
    start_time = time.time()
    w,a,mu,sigma = pf.mixture_fit(x,3,logger=logger,verbose=False,output_folder=output_folder,suffix=suffix)
    if logger is not None:
        logger.info('## mixture fit infomartion ##')
        logger.info('# Slope w=%0.3f: a=(%0.3f,%0.3f), '%(w[0],a[0],a[1]))
        for i in range(mu.shape[0]):
            logger.info('# Bump%d w=%0.3f:'%((i+1),w[i+1]))
            logger.info('  mu='+str(mu[i,:]))
            logger.info('  sigma='+str(sigma[i,:]))
        logger.info('Total time: %0.3fs\n'%(time.time()-start_time))

def main():   
    ## gradient check
    start_time = time.time()
    x,x_p,p = dl.load_toy_mixture(opt=2,logger=logger)
    _ = pf.ML_bump(x,logger=logger,gradient_check=True)
    logger.info('Total time: %0.3fs\n'%(time.time()-start_time))
    
    ## check ML_slope
    start_time = time.time()
    logger.info('## check ML_slope')
    sample,x,p = dl.load_toy_mixture(opt=0,logger=logger)
    a = pf.ML_slope(sample,c=0)
    logger.info('# fitted slope parameter')    
    logger.info('# a=(%0.4f,%0.4f)'%(a[0],a[1]))    
    logger.info('Total time: %0.3fs\n'%(time.time()-start_time))
    
    ## check f_slope
    start_time = time.time()
    logger.info('## check f_slope')
    a = np.array([2])
    x = np.array([[0.1],[0.5]])   
    p_true = 2/(np.exp(2)-1)*np.exp(x*a)
    logger.info('# True probability')
    logger.info('# p=(%0.4f,%0.4f)'%(p_true[0],p_true[1]))
    p = pf.f_slope(x,a)
    logger.info('# Fitted probability')
    logger.info('# p=(%0.4f,%0.4f)'%(p[0],p[1]))
    logger.info('Total time: %0.3fs\n'%(time.time()-start_time))
    
    ## check ML_bump
    start_time = time.time()
    logger.info('## check ML_bump')
    sample,x,p = dl.load_toy_mixture(opt=1,logger=logger)
    mu,sigma = pf.ML_bump(sample)
    logger.info('# fitted bump parameter')    
    logger.info('# mu=(%0.4f,%0.4f)'%(mu[0],mu[1]))
    logger.info('# sigma=(%0.4f,%0.4f)'%(sigma[0],sigma[1]))
    logger.info('Total time: %0.3fs\n'%(time.time()-start_time))
    
    ## check f_bump
    start_time = time.time()
    logger.info('## check f_bump')
    mu = np.array([0.2])
    sigma = np.array([0.2])
    x = np.array([[0.1],[0.5]])   
    p_true = norm.pdf(x,loc=0.2,scale=0.2)
    pmf = norm.cdf(1,loc=0.2,scale=0.2)-norm.cdf(0,loc=0.2,scale=0.2)
    p_true /= pmf
    logger.info('# True probability')
    logger.info('# p=(%0.4f,%0.4f)'%(p_true[0],p_true[1]))
    p = pf.f_bump(x,mu,sigma)
    logger.info('# Fitted probability')
    logger.info('# p=(%0.4f,%0.4f)'%(p[0],p[1]))
    logger.info('Total time: %0.3fs\n'%(time.time()-start_time))

    ## 2d_slope_bump  
    x,x_p,p = dl.load_toy_mixture(opt=2,logger=logger)
    suffix = '_2d_bump_slope'
    benchmark_mixture_fit(x,output_folder,suffix,logger=logger)
    
    ## 10d_slope_bump
    x,x_p,p = dl.load_toy_mixture(opt=3,logger=logger)
    suffix = '_10d_bump_slope'
    benchmark_mixture_fit(x,output_folder,suffix,logger=logger)
       
if __name__ == '__main__':
    main()