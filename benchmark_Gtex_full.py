import matplotlib
matplotlib.use('Agg')
import logging
import os
import data_loader as dl
import prim_fdr as pf
import time
import matplotlib.pyplot as plt

def main():
    ## create a new directory
    output_folder = './results/result_GTEx_full'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    logging.basicConfig(level=logging.DEBUG,
        format='%(module)s:: %(message)s',
        filename=output_folder+'/result.log', filemode='w')
    logger = logging.getLogger()
    
    ## loading the data 
    p,x,n_full,cate_name = dl.load_GTEx_full(verbose=True)

    ## report the baseline   
    n_rej,t_rej=pf.bh(p,alpha=0.1,n_sample=n_full,verbose=True)
    logger.info('## BH, n_rej=%d, t_rej=%0.5f'%(n_rej,t_rej))
    n_rej,t_rej,pi0_hat=pf.storey_bh(p,alpha=0.1,n_sample=n_full,lamb=0.995,verbose=True)
    logger.info('## SBH, n_rej=%d, t_rej=%0.5f'%(n_rej,t_rej))
    
    ## run the algorithm
    start_time = time.time()
    n_rej,t,_=pf.PrimFDR(p,x,5,alpha=0.1,h=None,n_itr=5000,verbose=True,output_folder=output_folder,logger=logger)
    logger.info('## PF, n_rej=%d'%(n_rej))
    logger.info('## Total time: %0.1fs'%(time.time()-start_time))
    
if __name__ == '__main__':
    main()