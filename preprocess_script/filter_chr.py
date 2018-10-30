import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', type = str, required = True)
    args = parser.parse_args()
    target = args.source + '.chr21.txt'

    fo = open(target, 'w')
    with open(args.source) as f:
        for line in f:
            if line.split(',')[0].split('-')[1].split('_')[0] == '21':
                fo.write(line)
    fo.close()

