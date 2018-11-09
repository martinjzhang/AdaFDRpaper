#!/usr/bin/env Rscript
library("mgcv")
library("adaptMT")
library('IHW')
library("splines")
library(swfdr)
args = commandArgs(trailingOnly=TRUE)
input_folder = args[1]
output_folder = args[2]
benchmark_bl = args[3]

print(paste0('input_folder: ', input_folder))
print(paste0('output_folder: ', output_folder))
print(paste0('benchmark_bl: ', benchmark_bl))

get_fdp_and_power <- function(h, h_hat){
  fdp <- sum((h==0)*(h_hat==1)) / sum(h_hat==1)
  power <- sum((h==1)*(h_hat==1)) / sum(h==1)
  res <- c(fdp, power)
  return(res)
}
# Create the output folder
if (!dir.exists(output_folder)){
  dir.create(output_folder)
}
# Set up the parameters.
alpha_list <- c(0.05, 0.1, 0.15, 0.2)
ind_list <- c(5, 10, 15, 20)
data_list <- list.files(path = input_folder)
rec_adapt <- list(fdp=c(), power=c(), alpha=c(), data_name=c())
rec_ihw <- list(fdp=c(), power=c(), alpha=c(), data_name=c())
rec_bl <- list(fdp=c(), power=c(), alpha=c(), data_name=c())
# Start processing simulation data
for (data_name in data_list){
  print(data_name)
  # Read data
  data_path <- paste0(input_folder, '/', data_name)
  temp_data <- read.table(data_path, sep = ',')
  p <- temp_data[, 1]
  h <- temp_data[, 2]
  x <- temp_data[, 3:ncol(temp_data)]
  # AdaptMT result
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
    # GAM
    formula <- 's(x1)'
    for (i in 2:ncol(temp_x)){
      formula <- paste0(formula, paste0('+s(x', i,')'))
    }
  }
  res_adapt <- adapt_gam(x = temp_x, pvals = p, pi_formulas = formula,
                         mu_formulas = formula, dist = dist, nfits = 5)
  for (i_alpha in 1:4){
    alpha <- alpha_list[i_alpha]
    h_hat <- integer(length(p))
    h_hat[res_adapt$rejs[[ind_list[i_alpha]]]] = 1
    res <- get_fdp_and_power(h, h_hat == 1)
    rec_adapt$fdp <- c(rec_adapt$fdp, res[1])
    rec_adapt$power <- c(rec_adapt$power, res[2])
    rec_adapt$alpha <- c(rec_adapt$alpha, alpha)
    rec_adapt$data_name <- c(rec_adapt$data_name, data_name)
  }
  # IHW result
  res_kmean <- kmeans(x, 20)
  x_1d <- as.factor(res_kmean$cluster)
  for (alpha in alpha_list){
    res_ihw <- ihw(p, x_1d, alpha)
    res <- get_fdp_and_power(h, adj_pvalues(res_ihw) <= alpha)
    rec_ihw$fdp <- c(rec_ihw$fdp, res[1])
    rec_ihw$power <- c(rec_ihw$power, res[2])
    rec_ihw$alpha <- c(rec_ihw$alpha, alpha)
    rec_ihw$data_name <- c(rec_ihw$data_name, data_name)
  }
  # BL result
  if (benchmark_bl == 'T'){
      res <- lm_pi0(p, X=x, smooth.df=5)
      p_adj <- res$pi0 * p.adjust(p, method = "BH") 
      for (alpha in alpha_list){
        res <- get_fdp_and_power(h, p_adj <= alpha)
        rec_bl$fdp <- c(rec_bl$fdp, res[1])
        rec_bl$power <- c(rec_bl$power, res[2])
        rec_bl$alpha <- c(rec_bl$alpha, alpha)
        rec_bl$data_name <- c(rec_bl$data_name, data_name)
      }
  }
}
# Write the result
rec_adapt <- data.frame(rec_adapt)
fname = paste0(output_folder, '/res_adapt')
write.table(rec_adapt, file=fname, row.names=TRUE, sep = ',')
rec_ihw <- data.frame(rec_ihw)
fname = paste0(output_folder, '/res_ihw')
write.table(rec_ihw, file=fname, row.names=TRUE, sep = ',')
rec_bl <- data.frame(rec_bl)
fname = paste0(output_folder, '/res_bl')
write.table(rec_bl, file=fname, row.names=TRUE, sep = ',')