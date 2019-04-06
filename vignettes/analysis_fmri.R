#!/usr/bin/env Rscript
library("mgcv")
library("adaptMT")
library('IHW')
library("splines")
library("swfdr")
args <- commandArgs(trailingOnly=TRUE)
data_path <- args[1]
alpha <- as.numeric(args[2])
print(paste0('data_path: ', data_path))
print(paste0('alpha: ', alpha))
print('Loading the data')
# Read data
temp_data <- read.table(data_path, sep = ',', header=TRUE)
p <- temp_data[, 1]
x <- temp_data[, 2]
# AdaptMT result
set.seed(0)
temp_x <- as.data.frame(x)
dist <- beta_family()
for (i in 1:ncol(temp_x)){
  colnames(temp_x)[i] <- paste0('x', i)
}
formula <- 's(x1)'
res_adapt <- adapt_gam(x = temp_x, pvals = p, pi_formulas = formula,
                       mu_formulas = formula, dist = dist, nfits = 5)
h_hat <- integer(length(p))
h_hat[res_adapt$rejs[[as.integer(alpha*100)]]] = 1
print(paste0('Number of rejections for AdaPT: ', sum(h_hat)))
# IHW result
set.seed(0)
res_ihw <- ihw(p, x, alpha)
print(paste0('Number of rejections for IHW: ', sum(adj_pvalues(res_ihw) <= alpha)))
# BL result
set.seed(0)
temp_x = as.matrix(x)
res <- lm_pi0(p, X=temp_x, smooth.df=5)
p_adj <- res$pi0 * p.adjust(p, method = "BH") 
p_adj_BH <- p.adjust(p, method = "BH") 
print(paste0('Number of rejections for BL: ', sum(p_adj <= alpha)))