## system settings 
import matplotlib
matplotlib.use('Agg')
import logging
import os
import sys
import argparse
import numpy as np

## nfdr2
import nfdr2.data_loader as dl
import nfdr2.method as md
import time
import matplotlib.pyplot as plt

p, x, h, n_full, _ = dl.load_2d_bump_slope(n_sample=20000)
_,t,_=md.method_hs(p,x,2,alpha=0.1,n_full=n_full,n_itr=50,verbose=True,
                   random_state=0, single_core=False, output_folder = './results')
FDP = np.sum((p < t)*(h == 0))/np.sum(p < t)
n_rej = np.sum(p < t)
print('n_rej', n_rej)
print('FDP', FDP)