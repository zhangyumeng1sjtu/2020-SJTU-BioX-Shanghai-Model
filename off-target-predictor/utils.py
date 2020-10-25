import numpy as np
import pandas as pd


match_info = {'match': [('A','A'), ('C','C'), ('G','G'), ('T','T')],
              'transition': [('A','T'), ('C','G'), ('G','C'), ('T','A')],
              'transversion': [('A','C'), ('A','G'), ('C','A'), ('C','T'),
                               ('G','A'), ('G','T'), ('T','C'), ('T','G')]}

def countGC(seq):
	count_dict = {bs: seq.count(bs) for bs in 'ATCG'}
	GCcontent = round((count_dict['G'] + count_dict['C'])/len(seq), 4)
	GCskew = round((count_dict['G'] - count_dict['C'])/(count_dict['G'] + count_dict['C']), 4) \
		if count_dict['G'] + count_dict['C'] != 0 else 0.0
	ATskew = round((count_dict['A'] - count_dict['T'])/(count_dict['A'] + count_dict['T']), 4) \
		if count_dict['A'] + count_dict['T'] != 0 else 0.0
	return GCcontent, GCskew, ATskew


def create_pmfm(data,label):
    pos_data = data[label==1]
    pmfm = np.zeros((3,20))
    for target, lure in pos_data.values:
        target = target[:20]
        lure = lure[:20]
        for j, (bs1, bs2) in enumerate(zip(target, lure)):
            for i, name in enumerate(match_info.keys()):
                if (bs1, bs2) in match_info[name]:
                    pmfm[i][j] += 0.5 if i==2 else 1 
                    break
    return pmfm/len(pos_data)


def feature_Encoding(df, target, lure, pmfm):
    target, lure = df[target], df[lure]
    target = target[:20]
    lure = lure[:20]
    global_feature = list(map(lambda x,y : x-y, countGC(target), countGC(lure)))
    pmfm_feature = np.zeros(20)
    for j, (bs1, bs2) in enumerate(zip(target, lure)):
        for i, name in enumerate(match_info.keys()):
            if (bs1, bs2) in match_info[name]:
                pmfm_feature[j] = pmfm[i][j]
                break
    
    return np.array(global_feature + list(pmfm_feature))
        

def create_Dataset(pos_csv, neg_csv):
    pos = pd.read_csv(pos_csv, header=None)
    pos = pd.DataFrame(pos, columns=[0,1])
    pos['label'] = 1
    neg = pd.read_csv(neg_csv, header=None)
    neg['label'] = -1
    return pd.concat([pos, neg])


def create_Input(data):
    target = np.array(data[0].values)
    lure = np.array(data[1].values)
    y = np.array(data['label'].values)
    X = pd.DataFrame({0:target,1:lure})
    return X,y