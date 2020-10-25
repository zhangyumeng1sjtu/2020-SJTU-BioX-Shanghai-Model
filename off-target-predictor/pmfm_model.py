from utils import *
from metrics import *
import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
from sklearn.metrics import auc, plot_roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


def k_fold_trainning(rawdata,n_folds=5):

    cv = StratifiedKFold(n_splits=n_folds,shuffle=True)
    target = np.array(rawdata[0].values)
    lure = np.array(rawdata[1].values)
    y = np.array(rawdata['label'].values)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(target, lure, y)):
        print('----------------Training Fold %d---------------'%(i+1))
        X_train = pd.DataFrame({0:target[train],1:lure[train]})
        X_test = pd.DataFrame({0:target[test],1:lure[test]})
        pmfm = create_pmfm(X_train,y[train])
        train_feature = X_train.apply(feature_Encoding, axis = 1, args = (0,1,pmfm)).values
        test_feature = X_test.apply(feature_Encoding, axis = 1, args = (0,1,pmfm)).values
        clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.9, max_depth=6, l2_regularization=100)
        train_data = np.matrix([train_feature[i] for i in range(train_feature.shape[0])])
        test_data = np.matrix([test_feature[i] for i in range(test_feature.shape[0])])
        clf.fit(train_data, y[train])
        pred = clf.predict(test_data)
        evaluate(y[test], pred)
        viz = plot_roc_curve(clf, test_data, y[test],
                            name='ROC fold {}'.format(i+1),
                            alpha=0.5, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver operating characteristic Curve")
    ax.legend(loc="lower right")
    plt.savefig('roc.png',dpi=300)


def train_model(rawdata, compute_importance=True):
    X,y = create_Input(rawdata)
    pmfm = create_pmfm(X,y)
    np.save("feature.npy",pmfm)
    feature = X.apply(feature_Encoding, axis = 1, args = (0,1,pmfm)).values
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.9, max_depth=6, l2_regularization=100)
    data = np.matrix([feature[i] for i in range(feature.shape[0])])
    clf.fit(data, y)
    joblib.dump(clf, "train_model.pkl")
    if compute_importance:
        feature_importance(clf, data, y)


def predict_offtarget(input_csv, model, pmfm, output):
    print('Predicting off-target sequences')
    data = pd.read_csv(input_csv, header=None)
    feature = data.apply(feature_Encoding, axis = 1, args = (0,1,pmfm)).values
    X = np.matrix([feature[i] for i in range(feature.shape[0])])
    clf = joblib.load(model)
    pred = clf.predict(X)
    data.loc[pred==1,2] = 'off-target'
    data.loc[pred==-1,2] = 'none-off-target'
    data.to_csv(output,index=False,header=False)


def feature_importance(clf, X ,y):
    result = permutation_importance(clf, X, y, n_repeats=5,random_state=42)
    sorted_idx = result.importances_mean.argsort()
    plt.style.use("seaborn-white")
    fig, ax = plt.subplots()
    feature_lables = np.array(['GC content','GC skew','AT skew'] + ["Position "+str(i) for i in range(1,21)])
    y = result.importances[sorted_idx].mean(axis=1).T
    ax.barh(feature_lables[sorted_idx],y,color="#87CEFA",yerr=result.importances_std[sorted_idx])
    for i, v in enumerate(y):
        ax.text(v + 0.0005, i-0.3, str(round(v,4)), color='blue', fontweight='bold')
    ax.set_title("Permutation Importances")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # fig.tight_layout()
    plt.savefig("importance.png",dpi=300)


def test_on_target(rawdata, sitename):
    print('------------Testing on %s-----------' % sitename)
    target_info = pd.read_csv("target_info.csv")
    if sitename in target_info['Site'].values:
        target_dict = target_info.set_index('Site').T.to_dict()
        sequence = target_dict[sitename]['Sequence']
        train_data = rawdata[rawdata[0]!=sequence]
        test_data = rawdata[rawdata[0]==sequence]
        X_train, y_train = create_Input(train_data)
        X_test, y_test = create_Input(test_data)
        pmfm = create_pmfm(X_train,y_train)
        train_feature = X_train.apply(feature_Encoding, axis = 1, args = (0,1,pmfm)).values
        test_feature = X_test.apply(feature_Encoding, axis = 1, args = (0,1,pmfm)).values
        clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.9, max_depth=6, l2_regularization=100)
        train_matrix = np.matrix([train_feature[i] for i in range(train_feature.shape[0])])
        test_matrix = np.matrix([test_feature[i] for i in range(test_feature.shape[0])])
        clf.fit(train_matrix, y_train)
        pred = clf.predict(test_matrix)
        evaluate(y_test, pred)
    else:
        print('ERROR: INCORRECT SITE NAME')