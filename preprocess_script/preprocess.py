import os
import numpy as np
from csv import reader
from config import *
import sys


selected = []
with open(metadata_path) as f:
    for line in reader(f):
        if line[-1] == 'TRUE':
            selected.append(line)

print(len(selected))

for item in selected:
    print(item[0][:-3])
#files that we need to use, I added to config.py


roadmap_dict = {}
for item in selected:
    roadmap_dict[item[0][:-3]] = item[2]


file_idx = int(sys.argv[1]) # which assiciation file to look at
association_file = os.path.join(association_path, files[file_idx])
#f = open(association_file)
#header = f.readline().strip().split()
#line = f.readline().strip().split()

print("processing file {}".format(association_file))

def get_expression_matrix(filename):
    with open(filename) as f:
        lines = f.readlines()
        head = lines[2]
        head = head.strip().split('\t')[2:]
        lines = lines[3:]

        expression_matrix = np.zeros((len(head), len(lines)))
        names = []
        for i, line in enumerate(lines):
            ls = line.strip().split('\t')
            names.append(ls[0])
            ls = np.array(list(map(float, ls[2:])))
            expression_matrix[:,i] = ls

        return expression_matrix, head, names

expression_matrix, head, gene_name = get_expression_matrix(os.path.join(utils_path, 'GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct'))
print(expression_matrix.shape, len(head), len(gene_name))
gene_idx = dict(zip(gene_name, range(len(gene_name))))
print(head)
tissue_idx = head.index(tissue_dict[files[file_idx]])

def process_line(line):
    gene = line[0]
    variant = line[1]
    pval = float(line[6])
    chromosome = variant.split('_')[0]
    dist = int(line[2])
    return gene, variant, pval, chromosome, dist

def get_gene_features(gene, expression_matrix, gene_idx):
    # get features of a gene
    expression = expression_matrix[gene_idx[gene]]
    return expression


snp_feat_dict = {}
with open(os.path.join(utils_path, 'snp_feat.txt')) as f:
    for line in f:
        ls = line.strip().split(',')
        snp_feat_dict[ls[0]] = ls[-1]

def get_snp_features(snp):
    # get features of a snp
    return snp_feat_dict.get(snp, 'NA')


tss_gene_dict = {}
with open(os.path.join(utils_path, 'tss.txt')) as f:
    ln = 0
    for line in f:
        if ln > 0:
            ls = line.strip().split(',')
            tss_gene_dict[ls[0]] = int(ls[-1])
        ln += 1


def get_pair_features(gene, snp):
    # get features of a gene-snp pair
    #print(gene)
    if '.' in gene:
        gene = gene.split('.')[0]

    if gene in tss_gene_dict:
        gene_pos = tss_gene_dict[gene]
    else:
        gene_pos = np.nan

    snp_pos = int(snp.split('_')[1])
    return  snp_pos - gene_pos


roadmap_file = roadmap_dict[files[file_idx]]
#print(roadmap_file)

roadmap_15state_fn = os.path.join(data_path, 'roadmap', roadmap_file + '_15_coreMarks_dense.bed')
roadmap_25state_fn = os.path.join(data_path, 'roadmap', roadmap_file + '_25_imputed12marks_dense.bed')

chrom_size = {}
with open(os.path.join(utils_path, 'hg19.chrom.sizes')) as f:
    for line in f:
        ls = line.strip().split('\t')
        chrom_size[ls[0]] = int(ls[1])

roadmap_15state = {}
ln = 0
with open(roadmap_15state_fn) as f:
    for line in f:
        if ln > 0:
            ls = line.strip().split('\t')
            if not ls[0] in roadmap_15state:
                roadmap_15state[ls[0]] = -1 * np.ones((chrom_size[ls[0]] // 100)).astype(np.int)

            roadmap_15state[ls[0]][int(ls[1])//100 : int(ls[2])//100] = int(ls[3].split('_')[0])

        ln += 1
        if ln % 100000 == 0:
            print(ln)


roadmap_25state = {}
ln = 0
with open(roadmap_25state_fn) as f:
    for line in f:
        if ln > 0:
            ls = line.strip().split('\t')
            if not ls[0] in roadmap_25state:
                roadmap_25state[ls[0]] = -1 * np.ones((chrom_size[ls[0]] // 100)).astype(np.int)

            roadmap_25state[ls[0]][int(ls[1]) //100 : int(ls[2]) // 100] = int(ls[3].split('_')[0])

        ln += 1
        if ln % 100000 == 0:
            print(ln)


def get_chromatin_state(chr_no, pos):
    if not 'chr' in chr_no:
        chr_no = 'chr{}'.format(chr_no)
    pos = int(np.floor(pos / 100))
    state_15 = roadmap_15state[chr_no][pos]
    state_25 = roadmap_25state[chr_no][pos]
    return state_15, state_25


def get_features_for_one_line(line):
    gene, variant, pval, chromosomem, tss_dist = process_line(line)
    expression = get_gene_features(gene, expression_matrix[tissue_idx], gene_idx)
    maf = get_snp_features(variant)

    if maf == 'NA':
        maf = 'nan'

    maf = float(maf)

    tss_dist2 = get_pair_features(gene, variant) #the tss_dist we calculated is 17 away from their annotation, so I
    #choose to use the tss_dist they provided
    #print(tss_dist, tss_dist2)

    vs =  variant.split('_')
    chrom_no = vs[0]
    pos = int(vs[1])


    chrom_state = get_chromatin_state(chrom_no, pos)
    return "{}-{}".format(gene, variant), expression, maf, tss_dist, chrom_state[0], chrom_state[1], pval


f = open(association_file)
header = f.readline().strip().split()
#print(header)
fo = open(association_file + '.processed_maf', 'w')
ln = 0
while True:
        line=f.readline()
        ln += 1
        if not line: break
        feats = get_features_for_one_line(line.strip().split())
        fo.write("{}, {}, {}, {}, {}, {}, {}\n".format(*feats))
        if ln % 10000 == 0:
            print("{} lines finished, {}%".format(ln, ln/(172.0 * 10**4)))

fo.close()
f.close()
