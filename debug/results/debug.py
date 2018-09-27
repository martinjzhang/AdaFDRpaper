## import public packages
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.stats import norm
from scipy.stats import multivariate_normal
import torch
from torch.autograd import Variable

## import self-written packages 
import nfdr2.method as md
import nfdr2.data_loader as dl

p, x, h, n_full, _ = dl.load_2d_bump_slope(n_sample=20000)
n_rej,t,_=md.method_hs(p,x,2,alpha=alpha,h=h,n_full=n_full,n_itr=50,verbose=True,\
                                output_folder='./results',random_state=0)