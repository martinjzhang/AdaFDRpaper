## system settings 
import matplotlib
matplotlib.use('Agg')
import logging
import os
import sys
import argparse
import adafdr.data_loader as dl
import adafdr.method as md
import time
import matplotlib.pyplot as plt
import pickle

def main(args):
    # Set up parameters.
    alpha = 0.1
    n_itr = 1500
    # Set up the output folder.
    output_folder = os.path.realpath('..') + '/result_small_data/result_' + args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        filelist = [os.remove(os.path.join(output_folder, f))\
                    for f in os.listdir(output_folder)]
    # Load the data.
    if 'GTEx' in args.data_loader:
        p,x,n_full,cate_name = dl.load_GTEx_full(verbose=True)
        x = x[:, 0:3]
        h = None
    else:
        p, h, x = eval('dl.'+args.data_loader+'()')
        n_full = p.shape[0]
    cate_name = {}
    # Logger.
    logging.basicConfig(level=logging.INFO,format='%(module)s:: %(message)s',\
                        filename=output_folder+'/result.log', filemode='w')
    logger = logging.getLogger()
    # An overview of the data
    logger.info('# p: %s'%str(p[0:2]))
    logger.info('# x: %s'%str(x[0:2, :]))
    # Report the baseline methods.
    n_rej, t_rej = md.bh_test(p, alpha=alpha, n_full=n_full, verbose=False)
    logger.info('## BH, n_rej=%d, t_rej=%0.5f'%(n_rej,t_rej))
    n_rej, t_rej, pi0_hat = md.sbh_test(p, alpha=alpha, n_full=n_full, verbose=False)
    logger.info('## SBH, n_rej=%d, t_rej=%0.5f, pi0_hat=%0.3f'%(n_rej, t_rej, pi0_hat))    
    # Analysis
    md.adafdr_explore(p, x, alpha=alpha, n_full=n_full, vis_dim=None, cate_name=cate_name,\
                       output_folder=output_folder, h=None)
    # Fast mode.
    output_folder_fast = output_folder + '_fast'
    if not os.path.exists(output_folder_fast):
        os.makedirs(output_folder_fast)
    else:
        filelist = [os.remove(os.path.join(output_folder_fast, f))\
                    for f in os.listdir(output_folder_fast)]
    logger.info('# p: %s'%str(p[0:2]))
    logger.info('# x: %s'%str(x[0:2, :]))
    start_time = time.time()
    n_rej,t_rej,_= md.adafdr_test(p, x, K=5, alpha=alpha, h=None, n_full=n_full, n_itr=n_itr,\
                                  verbose=True, output_folder=output_folder_fast, random_state=0,\
                                  fast_mode=True)
    logger.info('## AdaFDR (fast mode), n_rej1=%d, n_rej2=%d, n_rej_total=%d'%(n_rej[0],n_rej[1],n_rej[0]+n_rej[1]))
    logger.info('## Total time (fast mode): %0.1fs'%(time.time()-start_time))
    # Full mode.
    logger.info('# p: %s'%str(p[0:2]))
    logger.info('# x: %s'%str(x[0:2, :]))
    start_time = time.time()
    n_rej,t_rej,_= md.adafdr_test(p, x, K=5, alpha=alpha, h=None, n_full=n_full, n_itr=n_itr,\
                                  verbose=True, output_folder=output_folder, random_state=0)
    logger.info('## AdaFDR, n_rej1=%d, n_rej2=%d, n_rej_total=%d'%(n_rej[0],n_rej[1],n_rej[0]+n_rej[1]))
    logger.info('## Total time: %0.1fs'%(time.time()-start_time))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Side-info assisted multiple hypothesis testing')
    parser.add_argument('-d', '--data_loader', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required = True)
    args = parser.parse_args()
    main(args)