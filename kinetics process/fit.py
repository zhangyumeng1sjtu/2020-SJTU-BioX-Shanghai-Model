import pandas as pd
import numpy as np 
from sko.GA import GA
from utils import encode, predict

rawdata = pd.read_csv('VEGFA_xCas9.csv', header=None)
rawdata.columns = ['target','lure','off-target']

bounds = ([-10, -10, 0, 0], [0, 0, 10, 10])

def fun(parms):
    X = encode(rawdata)
    Y = rawdata['off-target'].values
    pred = predict(X, parms)
    return pred

def func(parms):
    X = encode(rawdata)
    Y = rawdata['off-target'].values
    pred = predict(X, parms)
    return np.sum(np.abs(pred-Y)) + 0.001*np.sum(parms)

ga = GA(func=func, n_dim=4, size_pop=100, max_iter=200, lb=bounds[0], ub=bounds[1])
best_x, best_y = ga.run()

print(best_x)

Y = rawdata['off-target'].values
pred = fun(best_x)
print(pred)
print(Y)
print(np.corrcoef(pred,Y)[0,1])
