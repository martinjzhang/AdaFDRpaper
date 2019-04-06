## Varying parameters 
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

# DATA_LOADER_LIST = ['load_GTEx']

def main(args):
    # Set up parameters.
    alpha = 0.1
    n_itr = 1500
    # Set up the output folder.
    output_folder = os.path.realpath('..') + '/result_simulation/result_' + args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        filelist = [os.remove(os.path.join(output_folder, f))\
                    for f in os.listdir(output_folder)]
    # Get logger.
    logging.basicConfig(level=logging.INFO,format='%(message)s',\
                        filename=output_folder+'/result.log', filemode='w')
    logger = logging.getLogger()
    # Run method in all data in the folder
    file_list = os.listdir(args.input_folder)
    alpha_list = [0.05]
    result_dic = {'bh': {}, 'sbh': {}, 'nfdr (fast)': {}, 'nfdr': {}}
    time_dic = {'nfdr (fast)': {}, 'nfdr': {}}
    for alpha in alpha_list:
        result_dic['bh'][alpha] = []
        result_dic['sbh'][alpha] = []
        result_dic['nfdr (fast)'][alpha] = []
        result_dic['nfdr'][alpha] = []
        time_dic['nfdr (fast)'][alpha] = {}
        time_dic['nfdr'][alpha] = {}
        for filename in file_list:
            filename_short = filename
            if filename[0] == '.':
                continue
            print('# Processing %s with alpha=%0.2f'%(filename, alpha))
            logger.info('# Processing %s with alpha=%0.2f'%(filename, alpha))
            filename = args.input_folder + '/' + filename
            p, x, h = dl.load_simulation_data(filename)
            n_full = p.shape[0]
            # Report the baseline.
            n_rej, t_rej = md.bh_test(p, alpha=alpha, n_full=n_full, verbose=False)
            result_dic['bh'][filename_short] = ([h, p<=t_rej])
            logger.info('## BH, n_rej=%d, t_rej=%0.5f'%(n_rej,t_rej))
            n_rej, t_rej, pi0_hat = md.sbh_test(p, alpha=alpha, n_full=n_full, verbose=False)
            result_dic['sbh'][filename_short] = [h, p<=t_rej]
            logger.info('## SBH, n_rej=%d, t_rej=%0.5f, pi0_hat=%0.3f'%(n_rej, t_rej, pi0_hat))
            # Fast mode.
            start_time = time.time()
            res = md.adafdr_test(p, x, K=5, alpha=alpha, h=h, n_full=n_full, n_itr=n_itr,\
                                 verbose=False, output_folder=None, random_state=0,\
                                 fast_mode=True)
            n_rej = res['n_rej']
            t_rej = res['threshold']
            time_dic['nfdr (fast)'][alpha][filename_short] = time.time()-start_time
            result_dic['nfdr (fast)'][filename_short] = [h, p<=t_rej]
            logger.info('## nfdr2 (fast mode), n_rej1=%d, n_rej2=%d, n_rej_total=%d'%(n_rej[0],n_rej[1],n_rej[0]+n_rej[1]))
            logger.info('## Total time (fast mode): %0.1fs'%(time.time()-start_time))
            # Full mode.
            start_time = time.time()
            res = md.adafdr_test(p, x, K=5, alpha=alpha, h=h, n_full=n_full, n_itr=n_itr,\
                                 verbose=False, output_folder=None, random_state=0,\
                                 fast_mode=False, single_core=False)
            n_rej = res['n_rej']
            t_rej = res['threshold']
            time_dic['nfdr'][alpha][filename_short] = time.time()-start_time
            result_dic['nfdr'][filename_short] = [h, p<=t_rej]
            logger.info('## nfdr2, n_rej1=%d, n_rej2=%d, n_rej_total=%d'%(n_rej[0],n_rej[1],n_rej[0]+n_rej[1]))
            logger.info('## Total time: %0.1fs'%(time.time()-start_time))
            logger.info('\n')
    # Store the result
    fil = open(output_folder+'/result_dic.pickle','wb') 
    pickle.dump(result_dic, fil)
    pickle.dump(time_dic, fil)
    fil.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Side-info assisted multiple hypothesis testing')
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required = True)
    parser.add_argument('-d', '--data_loader', type=str, required=False)
    parser.add_argument('-n', '--data_name', type=str, required=False)
    args = parser.parse_args()
    main(args)