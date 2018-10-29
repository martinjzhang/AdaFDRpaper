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
    output_folder = os.path.realpath('..') + '/result_gtex_feature_explore/result_'\
                    + args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        filelist = [os.remove(os.path.join(output_folder, f))\
                    for f in os.listdir(output_folder)]
    # Load the data.
    p, x, n_full, cate_name, cis_name = dl.load_GTEx(args.data_name,\
                                                     if_impute=False)
    # feature_explore
    md.feature_explore(p, x, alpha=alpha, n_full=n_full, vis_dim=None, cate_name=cate_name,\
                       output_folder=output_folder, h=None)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Side-info assisted multiple hypothesis testing')
    parser.add_argument('-d', '--data_loader', type=str, required=True)
    parser.add_argument('-n', '--data_name', type=str, required=False)
    parser.add_argument('-o', '--output_folder', type=str, required = True)
    args = parser.parse_args()
    main(args)