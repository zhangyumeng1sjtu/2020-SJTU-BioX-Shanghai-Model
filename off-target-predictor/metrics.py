from sklearn.metrics import accuracy_score,f1_score,roc_auc_score, \
    recall_score,precision_score,matthews_corrcoef, confusion_matrix
import numpy as np

def Sensitivity_Specificity(gt, pred): 
    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Sensitivity: %.3f ' % (float(TP) / float(TP+FN)))
    print('Specificity: %.3f ' % (float(TN) / float(TN+FP))) 


def evaluate(y, pred):
    print("Accuracy: %.3f " % accuracy_score(y, pred))
    Sensitivity_Specificity(y, pred)
    print("Precision: %.3f " % precision_score(y, pred))
    print("F-measure: %.3f " % f1_score(y, pred))
    print("MCC: %.3f " % matthews_corrcoef(y, pred))
    print("AUC: %.3f " % roc_auc_score(y, pred))
