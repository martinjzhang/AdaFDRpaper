## system settings 
import matplotlib
matplotlib.use('Agg')
import logging
import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
import pickle
import csv
import numpy as np

def main():
    n_full = 0
    start_time = time.time()
    # Process the MuTHER data
    input_folder = '/data3/martin/gtex_data/MuTHER'
    f_output_path = input_folder + '/' + 'MuTHER_cis_results_chrall.txt'
    chr_list = list(range(1, 24))
    # chr_list = [1]
    for i_chr in chr_list:
        filename = 'MuTHER_cis_results_chr%s.txt'%str(i_chr)        
        print(filename)
        f_path = input_folder + '/' + filename
        temp_data = np.loadtxt(f_path, delimiter='\t', skiprows=1, dtype=str)
        n_full = n_full + temp_data.shape[0]
        temp_data[:, 0] = i_chr
        temp_data = temp_data[:, [0,1,4,11,14,17]]
        if i_chr == chr_list[0]:
            with open(f_output_path, 'w') as f:
                csv.writer(f).writerows(temp_data)
        else:
            with open(f_output_path, 'a') as f:
                csv.writer(f).writerows(temp_data)
    print('n_full=%d, Completed, time:%0.1f'%(n_full, time.time()-start_time))
    f = open(input_folder + '/' + 'MuTHER_stats.txt', 'w')
    f.write('n_full = %d\n'%(n_full))
    f.write('time=%0.1fs\n'%(time.time()-start_time))
    f.write('Header\n')
    f.write('i_chr\tGene\tSNP\tFat_p\tLCL_p\tSkin_p\n')
    f.close()
    
if __name__ == '__main__':
    main()