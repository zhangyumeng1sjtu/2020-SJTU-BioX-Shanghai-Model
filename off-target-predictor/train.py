from pmfm_model import *
from utils import create_Dataset

data = create_Dataset('positive.csv', 'negative.csv')
k_fold_trainning(data, n_folds=5)

train_model(data,compute_importance=True)
