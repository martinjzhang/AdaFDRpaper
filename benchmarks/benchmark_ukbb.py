import matplotlib
matplotlib.use('Agg')
import logging
import os
import nfdr2.data_loader as dl
import nfdr2.prim_fdr as pf
import time
import matplotlib.pyplot as plt

import sys
import argparse


def main(args):
    output_folder = args.outf


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(module)s:: %(message)s',
                        filename=output_folder + '/result.log', filemode='w')
    logger = logging.getLogger()


    ## loading the data
    p, x, n_full, cate_name = dl.load_common_dataset(filename=args.file, n=args.n)

    nominal_alpha = 0.2

    ## report the baseline
    n_rej, t_rej = pf.bh(p, alpha=nominal_alpha, n_sample=n_full, verbose=True)
    logger.info('## BH, n_rej=%d, t_rej=%0.5f' % (n_rej, t_rej))
    n_rej, t_rej, pi0_hat = pf.storey_bh(p, alpha=nominal_alpha, n_sample=n_full, lamb=0.995, verbose=True)
    logger.info('## SBH, n_rej=%d, t_rej=%0.5f' % (n_rej, t_rej))

    ## run the algorithm
    start_time = time.time()
    n_rej, t, _ = pf.PrimFDR(p, x, 5, alpha=nominal_alpha, h=None, n_itr=1000, verbose=True, output_folder=output_folder,
                             logger=logger)
    logger.info('## PF, n_rej=%d' % (n_rej))
    logger.info('## Total time: %0.1fs' % (time.time() - start_time))

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='NeuralFDR2 multiple hypothesis testing')

    parser.add_argument('--file', type=str, required = True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--outf', type=str, required = True)

    args = parser.parse_args()
    main(args)