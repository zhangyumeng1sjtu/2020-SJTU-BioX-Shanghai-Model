import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

def compute_mismatch(csv,length=20):
    data = pd.read_csv(csv, header=None)
    mismatch = np.zeros(length)
    for target, lure in zip(data[0], data[1]):
        mismatch = compare_seq(target, lure, length, mismatch)
    return mismatch/len(data)


def compare_seq(target, lure, length, mismatch):
    for pos in range(length):
        if target[pos] != lure[pos]:
            mismatch[pos] += 1
    return mismatch
 
    
def plot_mismatch(pos, neg):
    pos_mismatch = compute_mismatch(pos)
    neg_mismatch = compute_mismatch(neg)
    x = np.arange(1,21)
    print(ttest_rel(pos_mismatch, neg_mismatch))
    pos_line = plt.plot(x,pos_mismatch,'r-',label='positive')
    neg_line = plt.plot(x,neg_mismatch,'b--',label='negative')
    plt.plot(x,pos_mismatch,'ro',x,neg_mismatch,'b^')
    plt.title('Position specific frequencies')
    plt.xlabel('Position(5\'->3\')')
    plt.ylabel('Mismatch frequency')
    plt.xticks([i for i in range(1,21)])
    plt.legend()
    plt.text(15,0.45,'paired t-test\np-value=0.033')
    plt.savefig('mismatch difference',dpi=300)


if __name__ == '__main__':
    plot_mismatch('positive.csv','negative.csv')

