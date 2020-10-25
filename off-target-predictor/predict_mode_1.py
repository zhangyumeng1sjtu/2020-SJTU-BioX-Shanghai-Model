from pmfm_model import predict_offtarget
import numpy as np
import time
import argparse
import os

parser = argparse.ArgumentParser(description='Find off-target sequences for target with NGG PAM')
parser.add_argument('-g','--genome_path',type=str,required=True, 
                            help="Input the abosulte path for the genome")
parser.add_argument('-t','--target_seq',type=str,required=True, 
                            help="Input the target sequence")
parser.add_argument('-n','--max_mismatch_num',type=str,default='6',choices=['3','4','5','6'], 
                            help="Input the maximum number of mismatch")
parser.add_argument('-f','--feature_file',type=str,default='feature.npy', 
                            help="Input the feature file")
parser.add_argument('-m','--model_file',type=str,default='train_model.pkl', 
                            help="Input the model file")                  
parser.add_argument('-o','--output_file',type=str,default='result', 
                            help="Output csv file name")                           
args = parser.parse_args()

with open('input.txt','w') as f:
    f.write('%s\nNNNNNNNNNNNNNNNNNNNNNGG\n%sNGG %s\n' % (args.genome_path,args.target_seq,args.max_mismatch_num))

print('Searching potential off-target sites on %s' % args.genome_path)
start = time.time()
os.system('cas-offinder input.txt G output.txt &>cas-offinder.log')
os.system('awk \'{print$1","$7}\' output.txt | sed \'s/a/A/g;s/c/C/g;s/g/G/g;s/t/T/g\' > temp.csv')
end = time.time()
print("It took %.2f seconds to collect all potential off-target sites." % (end-start))

start = time.time()
pmfm = np.load(args.feature_file)
predict_offtarget('temp.csv', args.model_file, pmfm, args.output_file +'.csv')
os.system('grep -v \'none-off-target\' %s.csv > off-target.csv' % args.output_file)
count = len(open('off-target.csv','r').readlines())
print('%d off-target sequence(s) for %s are found.' % (count, args.target_seq))
end = time.time()
print("It took %.2f seconds to obtain the off-target sequence(s)." % (end-start))
os.system('rm *txt')
os.system('rm temp.csv')
