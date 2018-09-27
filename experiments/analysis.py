## system settings 
import matplotlib
matplotlib.use('Agg')
import logging
import os
import sys
import argparse
import nfdr2.data_loader as dl
import nfdr2.method as md
import time
import matplotlib.pyplot as plt

DATA_LOADER_LIST = ['load_GTEx_small',\
                    'load_GTEx_Adipose_Subcutaneous',
                    'load_GTEx_Colon_Sigmoid',
                    'load_gtex_Artery_Aorta']

def main(args):
    # Set up parameters.
    alpha = 0.1
    n_itr = 1500
    # Set up the output folder
    output_folder = os.path.realpath('..') + '/results/result_' + args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        filelist = [os.remove(os.path.join(output_folder, f))\
                    for f in os.listdir(output_folder)]   
    # Load the data.
    if args.data_loader in DATA_LOADER_LIST:
        p, x, n_full, cate_name = eval('dl.'+args.data_loader+'()')
    else: 
        return
    # Logger.
    logging.basicConfig(level=logging.INFO,format='%(module)s:: %(message)s',\
                        filename=output_folder+'/result.log', filemode='w')
    logger = logging.getLogger()
    # Report the baseline methods.
    n_rej, t_rej = md.bh(p, alpha=alpha, n_full=n_full, verbose=False)
    logger.info('## BH, n_rej=%d, t_rej=%0.5f'%(n_rej,t_rej))
    n_rej, t_rej, pi0_hat = md.storey_bh(p, alpha=alpha, n_full=n_full, verbose=False)
    logger.info('## SBH, n_rej=%d, t_rej=%0.5f, pi0_hat=%0.3f'%(n_rej, t_rej, pi0_hat))
    # Analysis
    md.feature_explore(p, x, alpha=alpha, n_full=n_full, vis_dim=None, cate_name=cate_name,\
                       output_folder=output_folder, h=None)
    start_time = time.time()
    n_rej,t,_= md.method_hs(p, x, K=5, alpha=alpha, h=None, n_full=n_full, n_itr=n_itr,\
                            verbose=True, output_folder=output_folder, random_state=0)
    logger.info('## nfdr2, n_rej1=%d, n_rej2=%d, n_rej_total=%d'%(n_rej[0],n_rej[1],n_rej[0]+n_rej[1]))
    logger.info('## Total time: %0.1fs'%(time.time()-start_time))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Side-info assisted multiple hypothesis testing')
    parser.add_argument('-d', '--data_loader', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required = True)
    args = parser.parse_args()
    main(args)