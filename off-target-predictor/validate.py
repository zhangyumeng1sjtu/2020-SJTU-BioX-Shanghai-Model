from pmfm_model import test_on_target
from utils import create_Dataset
import pandas as pd

data = create_Dataset('positive.csv', 'negative.csv')
target_info = pd.read_csv("target_info.csv")

for sitename in target_info['Site'].values:
    test_on_target(data, sitename)