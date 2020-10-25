import numpy as np
from numpy.core.fromnumeric import cumsum
import pandas as pd 

def mismatch(pos_list):
    for i, pos in enumerate(pos_list):
        init = np.ones(20)
        if pos:
            init[list(pos)] = 0
            rev  = ~init.astype(np.bool)
        else:
            rev = np.zeros(20)
        temp = np.concatenate((init,rev.astype(np.int)),axis=0).reshape(2,20)
        temp = temp[np.newaxis,:]
        out = temp if i == 0 else np.concatenate((out,temp),axis=0)
    return out

def encode(data):
    for i in range(len(data)):
        target, lure = data['target'][i], data['lure'][i]
        temp = compseq(target, lure)
        out = temp if i == 0 else np.concatenate((out,temp),axis=0)
    return out


def compseq(seq1, seq2, spacer_len=20):
    target = seq1[:spacer_len]
    target = target[::-1]
    lure = seq2[:spacer_len]
    lure = lure[::-1]
    output = np.zeros((2,spacer_len))
    for i in range(spacer_len):
        bs1, bs2 = target[i], lure[i]
        if bs1 == bs2:
            output[0,i] = 1
        else:
            output[1,i] = 1
    return output[np.newaxis, :]


def predict(X, parms):
    pred = []
    for i in range(X.shape[0]):
        temp = np.array(parms[1:3])
        denominator = 1+np.exp(parms[0]+np.sum(temp.dot(X[i]))+parms[3])
        for j in range(1,X.shape[-1]):
            denominator += np.exp(parms[0]+np.sum(temp.dot(X[i][:,:j])))
        pred.append(1/denominator)
    return pred


def predict_progress(X, parms, spacer_len=20):
    energy = np.zeros((X.shape[0],spacer_len+3))
    for i in range(X.shape[0]):
        temp = np.array(parms[1:3])
        for j in range(1,spacer_len+3):
            if j == 1:
                energy[i,j] = parms[0]
            elif j == spacer_len+2:
                energy[i,j] = parms[3]
            else:
                energy[i,j] = temp.dot(X[i][:,j-2])
        energy[i] = cumsum(energy[i])
    return energy


def get_corrcoef(data, parms):
    data = pd.read_csv('VEGFA_xCas9.csv', header=None)
    data.columns = ['target','lure','off-target']
    X = encode(data)
    Y = data['off-target'].values
    pred = predict(X, parms)
    print(print(np.corrcoef(pred,Y)[0,1]))
