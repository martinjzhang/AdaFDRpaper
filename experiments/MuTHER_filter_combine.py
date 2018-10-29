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
import csv
import numpy as np

"""To do list:
    1. Change the naming to be consistent with gtex
"""
def main():
    n_full = 0
    n_fil = 0
    n_unmatch = 0
    start_time = time.time()
    # # Load the gene and snp information
    # # snp
    # fil_path = '/data3/martin/gtex_data/gtex_utils/snp_feat.txt'
    # snp_data = np.loadtxt(fil_path, delimiter=',', dtype=str)
    # snp_id2sym = {}
    # for i in range(snp_data.shape[0]):
    #     snp_id2sym[snp_data[i, 1]] = snp_data[i, 0]
    # # gene names
    # fil_path = '/data3/martin/gtex_data/gtex_utils/gencode.v19.genes.patched_contigs.gtf'
    # fil_open = open(fil_path, "r")
    # gene_sym2id = {}
    # for i_line,line in enumerate(fil_open):
    #     if line[0] != '#':
    #         line = line.strip().split('\t')
    #         line = line[8].strip().split(' ')
    #         gene_id = line[1].replace('"', '').replace(';', '')
    #         gene_name = line[9].replace('"', '').replace(';', '')
    #         gene_sym2id[gene_name] = gene_id
    # fil_open.close()
    # Process the MuTHER data
    input_folder = '/data3/martin/gtex_data/MuTHER'
    f_output_path = input_folder + '/' + 'MuTHER_cis_results_chrall.txt'
    chr_list = list(range(1, 24))
    for chr in chr_list:
        filename = 'MuTHER_cis_results_chr%s.txt'%str(chr)        
        print(filename)
        f_path = input_folder + '/' + filename
        temp_data = np.loadtxt(f_path, delimiter='\t', skiprows=1, dtype=str)
        n_full = n_full + temp_data.shape[0]
        p = np.array(temp_data[:, 11], dtype=float)
        # temp_data = temp_data[p<0.05, :]
        n_fil = n_fil + temp_data.shape[0]
        temp_data[:, 0] = chr
        temp_data = temp_data[:, [0,1,4,11]]
        # # Convert the name
        # ind_match = np.zeros([temp_data.shape[0]], dtype=bool)
        # for i in range(temp_data.shape[0]):
        #     gene_name = temp_data[i, 1]
        #     snp_id = temp_data[i, 2]
        #     if (gene_name in gene_sym2id) and (snp_id in snp_id2sym):
        #         temp_data[i, 1] = gene_sym2id[gene_name] + '-' + snp_id2sym[snp_id]
        #         ind_match[i] = True
        # temp_data = temp_data[ind_match, :]
        # temp_data = temp_data[:, [0,1,3]]
        # n_unmatch = n_unmatch + np.sum(ind_match==False)
        if chr == chr_list[0]:
            with open(f_output_path, 'w') as f:
                csv.writer(f).writerows(temp_data)
        else:
            with open(f_output_path, 'a') as f:
                csv.writer(f).writerows(temp_data)
    print('n_full=%d, Completed, time:%0.1f'%(n_full, time.time()-start_time))
    f = open(input_folder + '/' + 'MuTHER_stats.txt', 'w')
    f.write('n_full = %d, n_fil=%d, n_unmatch=%d'%(n_full, n_fil, n_unmatch))
    f.close()
    
if __name__ == '__main__':
    main()