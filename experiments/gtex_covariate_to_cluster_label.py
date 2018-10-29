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
def preprocess_gtex(x_input):
    x = np.copy(x_input)
    for i in range(x.shape[1]):
        ind_nan = np.isnan(x[:, i])
        if np.sum(~ind_nan) > 1:
            x[ind_nan, i] = np.mean(x[~ind_nan, i])
        else:
            x[ind_nan, i] = 0
    x[:, 0] = np.log10(x[:, 0]+0.5)
    return x

def main(args):
    np.random.seed(0)
    n_pretrain = 500000
    n_batch = 10000
    proportion_small = 0.01
    data_name = args.data_name
    file_input = '/data3/martin/gtex_data/GTEx_Analysis_v7_eQTL_all_associations/' +\
                 '%s.allpairs.txt.processed'%data_name
    file_output = '/data3/martin/nfdr2_simulation_data/gtex_cluster_data/' +\
                 '%s.allpairs.txt.processed'%data_name
    file_output_small = '/data3/martin/nfdr2_simulation_data/gtex_cluster_data_small/' +\
                        '%s.allpairs.txt.processed'%data_name
    print(data_name)
    print('Parameters: n_pretrain=%d, n_batch=%d, proportion_small=%0.2f'%\
          (n_pretrain, n_batch, proportion_small))
    print('file_input: %s'%file_input)
    print('file_output: %s'%file_output)
    print('file_output_small: %s'%file_output_small)
    start_time = time.time()
    # Pretrain the clustering model
    p = np.zeros([n_pretrain])
    x = np.zeros([n_pretrain, 4])
    f_input = open(file_input, 'r')
    i_pretrain = 0
    for i_line,line in enumerate(f_input):
        if np.random.rand(1)[0] < 0.01:
            line = line.strip().split(', ')
            x[i_pretrain] = line[1:5]
            p[i_pretrain] = line[-1]            
            if i_pretrain == n_pretrain-1:
                break
            i_pretrain = i_pretrain+1
    f_input.close()
    x = preprocess_gtex(x)
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x-x_mean)/x_std
    kmeans_pretrain = KMeans(n_clusters=20, random_state=0).fit(x)
    # Writing the clustering data
    f_input = open(file_input, 'r')
    f_output = open(file_output, 'w')
    f_output_small = open(file_output_small, 'w')
    for i_line,line in enumerate(f_input):
        if i_line%n_batch == 0:
            if i_line > 0:
                x = preprocess_gtex(x)
                x = (x-x_mean)/x_std
                x_label = kmeans_pretrain.predict(x)
                for i_hypothesis in range(len(p_value_list)):
                    csv_str = '%s, %s, %d\n'%(cis_name_list[i_hypothesis],\
                                              p_value_list[i_hypothesis],\
                                              x_label[i_hypothesis])
                    f_output.write(csv_str)
                    if np.random.rand(1)[0] <= proportion_small:
                        f_output_small.write(csv_str)
            cis_name_list = []
            p_value_list = []
            x = np.zeros([n_batch, 4], dtype=float)
        line = line.strip().split(', ')
        cis_name_list.append(line[0])
        p_value_list.append(line[-1])
        x[i_line%n_batch,:] = line[1:5]
        # print(i_line, line)
        # if i_line > 100:
        #     break
    # write the last few hypotheses
    x = preprocess_gtex(x)
    x = (x-x_mean)/x_std
    x_label = kmeans_pretrain.predict(x)
    for i_hypothesis in range(len(p_value_list)):
        csv_str = '%s, %s, %d\n'%(cis_name_list[i_hypothesis],\
                                  p_value_list[i_hypothesis],\
                                  x_label[i_hypothesis])
        f_output.write(csv_str)
        if np.random.rand(1)[0] <= proportion_small:
            f_output_small.write(csv_str)
    f_input.close()
    f_output.close()
    f_output_small.close()
    print('n_full=%d, Completed, time:%0.1f'%(i_line, time.time()-start_time))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Side-info assisted multiple hypothesis testing')
    parser.add_argument('-d', '--data_name', type=str, required=True)
    args = parser.parse_args()
    main(args)