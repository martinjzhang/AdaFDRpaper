import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', type = str, required = True)
    parser.add_argument('--second_tissue', type=str, required = True)
    args = parser.parse_args()
    target = args.source + '.augmented_not_related.txt'

    pv_dict = {}
    with open(args.second_tissue) as f:
        i = 0
        for line in f:
            ls = line.strip().split()
            pv_dict[ls[0]] = float(ls[-1])
            i += 1
            #if i > 1000:
            #    break

    fo = open(target, 'w')
    with open(args.source) as f:
        for line in f:
            name= line.strip().split()[0]
            fo.write(line.strip() + ", {}\n".format(pv_dict.get(name, float('nan'))))
    fo.close()

