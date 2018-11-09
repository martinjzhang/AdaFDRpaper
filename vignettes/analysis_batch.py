## system settings 
import numpy as np
import logging
import os
import sys
import argparse
import adafdr.data_loader as dl
import adafdr.method as md
import time
import pickle

def get_fdp_and_power(h, h_hat):
    fdp = np.sum((h==0)&(h_hat==1))/np.sum(h_hat==1)
    power = np.sum((h==1)&(h_hat==1)) / np.sum(h==1)
    return fdp, power
    
def main(args):    
    # Set up the parameters.
    input_folder = args.input_folder
    output_folder = './temp_result/res_' + args.data_name
    if args.alpha is not None:
        alpha_list = [args.alpha]
    else:
        alpha_list = [0.05, 0.1, 0.15, 0.2]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        filelist = [os.remove(os.path.join(output_folder, f))\
                    for f in os.listdir(output_folder)]
    print('input_folder: %s'%input_folder)
    print('output_folder: %s'%output_folder)
    print('alpha_list: %s'%alpha_list)
    # Get a file for recording.
    f_write = open(output_folder+'/result.log', 'w')
    # Process all data in the folder
    file_list = os.listdir(args.input_folder)   
    result_dic = {'bh': [], 'sbh': [], 'adafdr-fast': [], 'adafdr': []}
    for filename in file_list:
        if filename[0] == '.':
            continue        
        file_path = args.input_folder + '/' + filename
        p, x, h = dl.load_simulation_data(file_path)
        for alpha in alpha_list:
            print('# Processing %s with alpha=%0.2f'%(filename, alpha))
            f_write.write('# Processing %s with alpha=%0.2f\n'%(filename, alpha))
            # BH result
            n_rej, t_rej = md.bh_test(p, alpha=alpha, verbose=False)
            fdp,power = get_fdp_and_power(h, p<=t_rej)
            result_dic['bh'].append([fdp, power, alpha, filename])
            f_write.write('## BH discoveries: %d, threshold=%0.3f\n'%(n_rej,t_rej))
            # SBH result
            n_rej, t_rej, pi0_hat = md.sbh_test(p, alpha=alpha, verbose=False)
            fdp,power = get_fdp_and_power(h, p<=t_rej)
            result_dic['sbh'].append([fdp, power, alpha, filename])
            temp = '## SBH discoveries: %d, threshold=%0.3f, pi0_hat=%0.3f\n'%(n_rej, t_rej, pi0_hat)
            f_write.write(temp)
            # AdaFDR-fast result
            start_time = time.time()
            res = md.adafdr_test(p, x, alpha=alpha, fast_mode=True)
            n_rej = res['n_rej']
            t_rej = res['threshold']
            fdp,power = get_fdp_and_power(h, p<=t_rej)
            result_dic['adafdr-fast'].append([fdp, power, alpha, filename])
            temp = '## AdaFDR-fast discoveries: fold_1=%d, fold_2=%d, total=%d\n'%\
                    (n_rej[0],n_rej[1],n_rej[0]+n_rej[1])
            f_write.write(temp)
            f_write.write('## Time: %0.1fs'%(time.time()-start_time))
            # AdaFDR result
            start_time = time.time()
            res = md.adafdr_test(p, x, alpha=alpha, fast_mode=False)
            n_rej = res['n_rej']
            t_rej = res['threshold']
            fdp,power = get_fdp_and_power(h, p<=t_rej)
            result_dic['adafdr'].append([fdp, power, alpha, filename])
            temp = '## AdaFDR discoveries: fold_1=%d, fold_2=%d, total=%d\n'%\
                    (n_rej[0],n_rej[1],n_rej[0]+n_rej[1])
            f_write.write(temp)
            f_write.write('## Time: %0.1fs'%(time.time()-start_time))
            f_write.write('\n')
    # Store the result
    fil = open(output_folder+'/result.pickle','wb') 
    pickle.dump(result_dic, fil)
    fil.close()   
    f_write.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Side-info assisted multiple hypothesis testing')
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-d', '--data_name', type=str, required=True)
    parser.add_argument('-a', '--alpha', type=float, required=False)
    args = parser.parse_args()
    main(args)