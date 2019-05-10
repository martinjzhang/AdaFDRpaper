## import public packages
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import ttest_ind
import time
import os
import pickle

## import self-written packages 
from adafdr.util import *
import adafdr.data_loader as dl
import adafdr.method as md

def standardize_cis_name(cis_name, gene_id2sym, snp_sym2id):
    cis_name_standard = []
    for i_name,name in enumerate(cis_name):
        gene_id,snp_sym = name.split('-')
        if (gene_id in gene_id2sym) and (snp_sym in snp_sym2id):
            cis_name_standard.append(gene_id2sym[gene_id]+'-'+snp_sym2id[snp_sym])
    return cis_name_standard

def get_MuTHER_p_value(cis_set, MuTHER_dic, GTEx_dic, gene_sym2id, snp_id2sym, fil_rec):
    p_MuTHER = np.ones([len(cis_set), 2], dtype=float)
    ct_unmatch=0
    ct_match = 0
    for i_name,name in enumerate(cis_set):
        if name in MuTHER_dic:
            # print(name)
            gene_sym, snp_id = name.split('-rs')
            snp_id = 'rs' + snp_id
            gtex_name = gene_sym2id[gene_sym]+'-'+snp_id2sym[snp_id]
            p_MuTHER[i_name, 0] = MuTHER_dic[name]
            if gtex_name in GTEx_dic:
                p_MuTHER[i_name, 1] = GTEx_dic[gtex_name]
                ct_match += 1
            else:
                ct_unmatch+=1
    fil_rec.write('ct_unmatch=%d, ct_match=%d\n'%(ct_unmatch, ct_match))
    return p_MuTHER

def main():
    # Load reference
    f_output = '/home/martin/NeuralFDR2/result_downstream_v1/recs_LCL'
    fil_rec = open(f_output,'w') 
    fil_rec.write('# Load snp and gene\n')
    # snp names
    fil_path = '/data3/martin/gtex_data/gtex_utils/snp_feat.txt'
    snp_data = np.loadtxt(fil_path, delimiter=',', dtype=str)
    snp_sym2id = {}
    snp_id2sym = {}
    for i in range(snp_data.shape[0]):
        snp_sym2id[snp_data[i, 0]] = snp_data[i, 1]
        snp_id2sym[snp_data[i, 1]] = snp_data[i, 0]
    # gene names
    fil_path = '/data3/martin/gtex_data/gtex_utils/gencode.v19.genes.patched_contigs.gtf'
    fil_open = open(fil_path, "r")
    gene_sym2id = {}
    gene_id2sym = {}
    for i_line,line in enumerate(fil_open):
        if line[0] != '#':
            line = line.strip().split('\t')
            line = line[8].strip().split(' ')
            gene_id = line[1].replace('"', '').replace(';', '')
            gene_name = line[9].replace('"', '').replace(';', '')
            gene_id2sym[gene_id] = gene_name
            gene_sym2id[gene_name] = gene_id
    fil_open.close()
    # Load MuTHER data
    fil_rec.write('# Load MuTHER\n')
    file_muther_path = '/data3/martin/gtex_data/MuTHER/' + 'MuTHER_cis_results_chrall.txt'
    data_muther = np.loadtxt(file_muther_path, delimiter=',', dtype=str)
    MuTHER_dic = {}
    count_na = 0
    for i in range(data_muther.shape[0]):
        gene_sym,snp_id = data_muther[i, [1,2]]
        if ('n' in data_muther[i, 4]) or ('N' in data_muther[i, 4]):
            count_na = count_na+1
        else:
            MuTHER_dic[gene_sym+'-'+snp_id] = float(data_muther[i, 4])
    fil_rec.write('# MuTHER_dic count_na=%d\n'%count_na)
    # for i in range(data_muther.shape[0]):
    #     gene_sym,snp_id = data_muther[i, [1,2]]
    #     MuTHER_dic[gene_sym+'-'+snp_id] = float(data_muther[i, 3])
    n_full = 29160396
    # p = np.array(data_muther[:,-2], dtype=float) # fixit
    # Process GTEx data
    fil_rec.write('# Process GTEx\n')
    # data_list = ['Adipose_Subcutaneous', 'Adipose_Visceral_Omentum', 
    #              'Cells_EBV-transformed_lymphocytes']
    data_list = ['Cells_EBV-transformed_lymphocytes']
    # data_list = ['Adipose_Subcutaneous_chr21']
    output_folder = '/home/martin/NeuralFDR2/result_downstream_v1'
    load_gtex_data_name_dic = {'Adipose_Subcutaneous':'Adipose_Subcutaneous',
                               'Adipose_Subcutaneous_chr21':'Adipose_Subcutaneous-chr21',
                               'Adipose_Visceral_Omentum':'Adipose_Visceral_Omentum',
                               'Adipose_Visceral_Omentum_chr21':
                               'Adipose_Visceral_Omentum-chr21',
                               'Cells_EBV-transformed_lymphocytes':
                               'Cells_EBV-transformed_lymphocytes'}
    for data_name in data_list:
        # Load results
        fil_rec.write('\n'+data_name+'\n')
        res_GTEx_path = '/data3/martin/gtex_data/results/' + \
                        'result_GTEx_%s.pickle'%data_name
        fil = open(res_GTEx_path,'rb') 
        result_dic = pickle.load(fil)
        fil.close()  
        h_hat_sbh = result_dic['sbh']['h_hat']
        h_hat_nfdr = result_dic['nfdr']['h_hat']
        fil_rec.write('# D_sbh=%d\n'%np.sum(h_hat_sbh))
        fil_rec.write('# D_nfdr=%d, D_overlap=%d\n'\
                      %(np.sum(h_hat_nfdr), np.sum(h_hat_sbh*h_hat_nfdr)))
        p_gtex,_,_,_,cis_name = dl.load_GTEx(load_gtex_data_name_dic[data_name])
        GTEx_dic = {}
        for i_cis_name in range(cis_name.shape[0]):
            GTEx_dic[cis_name[i_cis_name]] = p_gtex[i_cis_name]
        cis_name_sbh = cis_name[h_hat_sbh]
        cis_name_nfdr = cis_name[h_hat_nfdr]
        # Standardize the name.
        cis_nfdr_standard = standardize_cis_name(cis_name_nfdr, gene_id2sym, snp_sym2id)
        cis_sbh_standard = standardize_cis_name(cis_name_sbh, gene_id2sym, snp_sym2id)
        # Look at the difference.     
        cis_nfdr_standard = set(cis_nfdr_standard)
        cis_sbh_standard = set(cis_sbh_standard)
        cis_intersect = cis_nfdr_standard & cis_sbh_standard
        cis_sbh = cis_sbh_standard - cis_intersect
        cis_nfdr = cis_nfdr_standard - cis_intersect
        # Compute the corresponding p-values
        p_MuTHER_intersect = get_MuTHER_p_value(cis_intersect, MuTHER_dic, GTEx_dic,
                                                gene_sym2id, snp_id2sym, fil_rec)
        p_MuTHER_sbh = get_MuTHER_p_value(cis_sbh, MuTHER_dic, GTEx_dic,
                                          gene_sym2id, snp_id2sym, fil_rec)
        p_MuTHER_nfdr = get_MuTHER_p_value(cis_nfdr, MuTHER_dic, GTEx_dic,
                                           gene_sym2id, snp_id2sym, fil_rec)
        # Save results
        n_counts = np.array([len(cis_intersect), len(cis_sbh), len(cis_nfdr)])
        fil = open(output_folder+'/p_overlap_%s.pickle'%data_name,'wb') 
        pickle.dump(n_counts, fil)
        pickle.dump(p_MuTHER_intersect[p_MuTHER_intersect[:, 0]<1, :], fil)
        pickle.dump(p_MuTHER_sbh[p_MuTHER_sbh[:, 0]<1, :], fil)
        pickle.dump(p_MuTHER_nfdr[p_MuTHER_nfdr[:, 0]<1, :], fil)
        fil.close()
    fil_rec.close()
    
if __name__ == '__main__':
    main()