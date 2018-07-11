import sys
import numpy as np

tree_file = sys.argv[1]
cat_code = sys.argv[2]

metadata = np.loadtxt(tree_file, skiprows=1, delimiter='\t', dtype=str)

toplevel = [item for item in metadata if item[-2] == '0']

# for each item in categories of top level, generate some experiments

for cat in toplevel:
    #print(cat)
    selected = [item for item in metadata if item[-2] == cat[-3] and item[-1] == "Y"]
    if len(selected) > 3:
        #print(selected)

        for item in selected:
            #print(item)
            selected_other = [it for it in selected if not it is item]
            #print(len(selected_other))
            l = len(selected_other)
            choice = np.random.randint(l, size=2)
            print("python process_ukbb.py {}_{}.assoc.tsv {}_{}.assoc.tsv {}_{}.assoc.tsv".format(cat_code, item[0], cat_code,
                                selected_other[choice[0]][0],  cat_code, selected_other[choice[1]][0]))
