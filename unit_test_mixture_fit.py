import matplotlib
matplotlib.use('Agg')
import logging
import os
import data_loader as dl
import prim_fdr as pf
import time
import matplotlib.pyplot as plt


def benchmark_mixture_fit(x,output_folder,suffix,logger=None):
    start_time = time.time()
    w,a,mu,sigma = pf.mixture_fit(x,3,logger=logger,verbose=False,output_folder=output_folder,suffix=suffix)
    if logger is not None:
        logger.info('## mixture fit infomartion ##')
        logger.info('# Slope w=%0.2f: a=(%0.1f,%0.1f), '%(w[0],a[0],a[1]))
        for i in range(mu.shape[0]):
            logger.info('# Bump%d w=%0.2f:'%((i+1),w[i+1]))
            logger.info('  mu='+str(mu[i,:]))
            logger.info('  sigma='+str(sigma[i,:]))
        logger.info('Total time: %0.1fs'%(time.time()-start_time))
        logger.info('\n')

def main():
    ## create a new directory
    output_folder = './results/unit_test_mixture_fit'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    ## have a logger 
    
    #logging.basicConfig(level=logging.DEBUG,
    #    format='%(asctime)s %(module)s:: %(message)s',
    #    filename=output_folder+'/result.log', filemode='w')
    logging.basicConfig(level=logging.DEBUG,
        format='%(module)s:: %(message)s',
        filename=output_folder+'/result.log', filemode='w')
    logger = logging.getLogger()
    
    ## gradient check
    x,x_p,p = dl.load_toy_mixture(opt=2,logger=logger)
    _ = pf.ML_bump(x,logger=logger,gradient_check=True)
    
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