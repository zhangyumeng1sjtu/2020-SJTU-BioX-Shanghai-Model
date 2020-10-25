import pandas as pd
import numpy as np

pos_data = pd.read_csv('positive.csv')
pos_data.columns = ['target','lure','count']
targets = np.unique(pos_data['target'].values)
pos_dict = {target: pos_data[pos_data['target']==target]['lure'].values for target in targets}

all = open('all.csv','r')
with open('negative.csv','w') as f:
    line = all.readline()
    while line:
        line = line.strip()
        target, lure = line.split(',')
        if lure not in pos_dict[target]:
            f.write(line+"\n")
        line = all.readline()
all.close()