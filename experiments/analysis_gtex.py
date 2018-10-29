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
import pickle

def main(args):
    # Set up parameters.
    alpha = 0.01
    n_itr = 1500
    # Set up the output folder.
    output_folder = os.path.realpath('..') + '/results/result_' + args.output_folder
    output_datafile = '/data3/martin/gtex_data/results/result_' + args.output_folder+'.pickle'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        filelist = [os.remove(os.path.join(output_folder, f))\
                    for f in os.listdir(output_folder)]
    # Load the data.
    p, x, n_full, cate_name, cis_name = dl.load_GTEx(args.data_name)
    # Logger.
    logging.basicConfig(level=logging.INFO,format='%(module)s:: %(message)s',\
                        filename=output_folder+'/result.log', filemode='w')
    logger = logging.getLogger()
    result_dic = {}
    # An overview of the data
    logger.info('# p: %s'%str(p[0:2]))
    logger.info('# x: %s'%str(x[0:2, :]))
    # Report the baseline methods.
    n_rej, t_rej = md.bh(p, alpha=alpha, n_full=n_full, verbose=False)
    logger.info('## BH, n_rej=%d, t_rej=%0.5f'%(n_rej,t_rej))
    result_dic['bh'] = {'h_hat': p < t_rej}
    n_rej, t_rej, pi0_hat = md.storey_bh(p, alpha=alpha, n_full=n_full, verbose=False)
    result_dic['sbh'] = {'h_hat': p < t_rej}
    logger.info('## SBH, n_rej=%d, t_rej=%0.5f, pi0_hat=%0.3f'%(n_rej, t_rej, pi0_hat))    
    # Analysis
    md.feature_explore(p, x, alpha=alpha, n_full=n_full, vis_dim=None, cate_name=cate_name,\
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
    n_rej,t_rej,_= md.method_hs(p, x, K=5, alpha=alpha, h=None, n_full=n_full, n_itr=n_itr,\
                                verbose=True, output_folder=output_folder_fast, random_state=0,\
                                fast_mode=True)
    result_dic['nfdr (fast)'] = {'h_hat': p < t_rej}
    logger.info('## nfdr2 (fast mode), n_rej1=%d, n_rej2=%d, n_rej_total=%d'%(n_rej[0],n_rej[1],n_rej[0]+n_rej[1]))
    logger.info('## Total time (fast mode): %0.1fs'%(time.time()-start_time))
    # Full mode.
    logger.info('# p: %s'%str(p[0:2]))
    logger.info('# x: %s'%str(x[0:2, :]))
    start_time = time.time()
    n_rej,t_rej,_= md.method_hs(p, x, K=5, alpha=alpha, h=None, n_full=n_full, n_itr=n_itr,\
                                verbose=True, output_folder=output_folder, random_state=0)
    result_dic['nfdr'] = {'h_hat': p < t_rej}
    logger.info('## nfdr2, n_rej1=%d, n_rej2=%d, n_rej_total=%d'%(n_rej[0],n_rej[1],n_rej[0]+n_rej[1]))
    logger.info('## Total time: %0.1fs'%(time.time()-start_time))
    # Store the result
    fil = open(output_datafile,'wb') 
    pickle.dump(result_dic, fil)
    fil.close()  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Side-info assisted multiple hypothesis testing')
    parser.add_argument('-d', '--data_loader', type=str, required=True)
    parser.add_argument('-n', '--data_name', type=str, required=False)
    parser.add_argument('-o', '--output_folder', type=str, required = True)
    args = parser.parse_args()
    main(args)