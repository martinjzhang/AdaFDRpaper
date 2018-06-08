## system settings 
import matplotlib
matplotlib.use('Agg')
import logging
import os

output_folder = os.path.realpath('..') + '/results/result_airway'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
logging.basicConfig(level=logging.INFO,format='%(module)s:: %(message)s',\
                    filename=output_folder+'/result.log', filemode='w')
logger = logging.getLogger()

## nfdr2
import nfdr2.data_loader as dl
import nfdr2.prim_fdr as pf
import time
import matplotlib.pyplot as plt

def main():    
    ## loading the data 
    p,x = dl.load_airway(verbose=True)

    ## report the baseline   
    n_rej,t_rej=pf.bh(p,alpha=0.1,verbose=False)
    logger.info('## BH, n_rej=%d, t_rej=%0.5f'%(n_rej,t_rej))
    n_rej,t_rej,pi0_hat=pf.storey_bh(p,alpha=0.1,verbose=False)
    logger.info('## SBH, n_rej=%d, t_rej=%0.5f'%(n_rej,t_rej))
    
    ## feature explore function
    #pf.feature_explore(p,x,alpha=0.1,qt_norm=False,vis_dim=None,cate_name={},output_folder=output_folder)
    
    ## run the algorithm
    start_time = time.time()
    n_rej,t,_=pf.PrimFDR_cv(p,x,5,alpha=0.1,h=None,n_itr=1500,verbose=True,\
                         output_folder=output_folder,logger=logger)
    logger.info('## nfdr2, n_rej=%d'%(n_rej))
    logger.info('## Total time: %0.1fs'%(time.time()-start_time))
    
if __name__ == '__main__':
    main()