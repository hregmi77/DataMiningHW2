import pandas as pd
import arff
import numpy as np
from numpy.random import randint
from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, roc_curve, auc
def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys
# Read Data
n_features = [20, 80, 140, 200, 260, 320]
n_trials = 50
best_auc_score = 0
traindatafull = pd.read_csv("CombinedTrain10000_Normalized.csv")
train_y = traindatafull['334']
trainfull_x = traindatafull.drop(columns='334')
trainfull_cols = trainfull_x.columns
for featuresize in n_features:
    for trials in range(n_trials):
        featureidx = randint(0, trainfull_cols.shape[0], featuresize) # Last Column is output
        trainx_cols = trainfull_cols[featureidx]
        train_x = trainfull_x[trainx_cols]
        # Reading and Filtering Testdata
        testdatafull = pd.read_csv("CombinedTest10000_Normalized.csv")
        test_y = testdatafull['334']
        testfull_x = testdatafull.drop(columns='334')
        test_x = testfull_x[trainx_cols]
        # Converting to numpy for faster computation
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        # Balancing the dataset
        train_bx, train_by = balanced_subsample(train_x, train_y)
        clf = RandomForestClassifier()
        clf.fit(train_bx, train_by)
        predicted_y = clf.predict(test_x)
        # Compute AUC Score
        auc_svm = roc_auc_score(test_y, predicted_y)
        if auc_svm > best_auc_score:
            best_auc_score = auc_svm
            print('Feature Size:', featuresize,'Trial Num:', trials, 'AUC Score:', auc_svm)
            best_colums = trainx_cols
print('Best Feature Indexs', best_colums, 'Best AUC Score:', best_auc_score)
