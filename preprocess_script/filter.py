import sys

fo = open(sys.argv[1] + '.filtered', 'w')
ln = 0
with open(sys.argv[1]) as f:
    for line in f:
        ln += 1
        pvalue = float(line.strip().split()[-1])
        if pvalue > 0.99 or pvalue < 0.01:
            fo.write(line)
            
            
fo.close()
print("{} {}".format(sys.argv[1], ln))