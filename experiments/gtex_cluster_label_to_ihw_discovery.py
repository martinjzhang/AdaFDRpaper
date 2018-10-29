## system settings 
## import public packages
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from sklearn.cluster import KMeans
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
import time
import os

## import self-written packages 
from nfdr2.util import *
import nfdr2.method as md
import nfdr2.data_loader as dl

"""To do list:
    1. Change the naming to be consistent with gtex
"""

def main(args):
    np.random.seed(0)
    data_name = args.data_name
    file_input = '/home/martin/NeuralFDR2/result_small_data/res_gtex_cluster_small/' +\
                 'threshold_%s.allpairs.txt.processed'%data_name
    file_input_gtex_cluster = '/data3/martin/nfdr2_simulation_data/gtex_cluster_data/' +\
                              '%s.allpairs.txt.processed'%data_name
    file_output = '/home/martin/NeuralFDR2/result_small_data/res_gtex_cluster_small/' +\
                 'result_%s'%data_name
    print(data_name)
    print('file_input: %s'%file_input)
    print('file_output: %s'%file_output)
    start_time = time.time()
    temp_data = np.loadtxt(file_input, skiprows=1, delimiter=',')
    threshold_dic = {}
    for i_row in range(temp_data.shape[0]):
        threshold_dic[temp_data[i_row, 0]-1] = temp_data[i_row, 1]
    f_input = open(file_input_gtex_cluster, 'r')
    f_output = open(file_output, 'w')
    n_rej = 0
    n_full = 0
    # print(threshold_dic)
    for i_line,line in enumerate(f_input):
        _, p, x = line.split(',')
        # print(_, p, x)
        if float(p) <= threshold_dic[int(x)]:
            n_rej = n_rej + 1
        n_full = n_full + 1        
        # if i_line > 100000:
        #     break
    output_str = '%s: n_full=%d, n_rej=%d, Completed, time:%0.1f'%\
                (data_name, n_full, n_rej, time.time()-start_time)
    f_output.write(output_str)
    print(output_str)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Side-info assisted multiple hypothesis testing')
    parser.add_argument('-d', '--data_name', type=str, required=True)
    args = parser.parse_args()
    main(args)