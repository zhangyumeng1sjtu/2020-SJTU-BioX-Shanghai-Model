from pmfm_model import predict_offtarget
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Find off-target sequences for target with NGG PAM')
parser.add_argument('-c','--csv_file',type=str,required=True, 
                            help="Input target and potential off-target sequence pairs in a .csv file")
parser.add_argument('-f','--feature_file',type=str,default='feature.npy', 
                            help="Input the feature file")
parser.add_argument('-m','--model_file',type=str,default='train_model.pkl', 
                            help="Input the model file")                  
parser.add_argument('-o','--output_file',type=str,default='result', 
                            help="Output csv file name")                           
args = parser.parse_args()

pmfm = np.load(args.feature_file)
predict_offtarget(args.csv_file, args.model_file, pmfm, args.output_file +'.csv')
os.system('grep -v \'none-off-target\' %s.csv > off-target.csv' % args.output_file)
count = len(open('off-target.csv','r').readlines())
print('%d off-target sequence(s) are found.' % count)
