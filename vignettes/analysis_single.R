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
x <- temp_data[, 2:ncol(temp_data)]
# AdaptMT result
set.seed(0)
temp_x <- as.data.frame(x)
dist <- beta_family()
for (i in 1:ncol(temp_x)){
  colnames(temp_x)[i] <- paste0('x', i)
}
if (ncol(temp_x) == 1){
  formula <- 's(x1)'
}else if (ncol(temp_x)==2){
  formula <- 's(x1)+s(x2)'
}else{
  formula <- 's(x1)'
  for (i in 2:ncol(temp_x)){
    formula <- paste0(formula, paste0('+s(x', i,')'))
  }
}
res_adapt <- adapt_gam(x = temp_x, pvals = p, pi_formulas = formula,
                       mu_formulas = formula, dist = dist, nfits = 5)
h_hat <- integer(length(p))
h_hat[res_adapt$rejs[[as.integer(alpha*100)]]] = 1
print(paste0('Number of rejections for AdaPT: ', sum(h_hat)))
# IHW result
set.seed(0)
if (ncol(temp_data)==2){
    res_ihw <- ihw(p, x, alpha)
} else{
    x <- scale(x, center = TRUE, scale = TRUE)
    res_kmean <- kmeans(x, 20)
    x_1d <- as.factor(res_kmean$cluster)
    res_ihw <- ihw(p, x_1d, alpha)
}
print(paste0('Number of rejections for IHW: ', sum(adj_pvalues(res_ihw) <= alpha)))
# BL result
set.seed(0)
temp_x = as.matrix(x)
res <- lm_pi0(p, X=temp_x, smooth.df=5)
p_adj <- res$pi0 * p.adjust(p, method = "BH") 
p_adj_BH <- p.adjust(p, method = "BH") 
print(paste0('Number of rejections for BL: ', sum(p_adj <= alpha)))
# print(paste0('Number of rejections for BH: ', sum(p_adj_BH <= alpha)))